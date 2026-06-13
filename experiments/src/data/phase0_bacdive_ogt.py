#!/usr/bin/env python3
"""
Phase 0: BacDive OGT Label Replacement

Queries BacDive for curated growth temperatures, replaces OGT labels
where available with a 20°C sanity gate to catch taxonomy mismatches.

Usage:
    python phase0_bacdive_ogt.py [--cache-only] [--dry-run]
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_FILE = PROJECT_ROOT / "data" / "bacdive_ogt_cache.json"
DATA_FILE = PROJECT_ROOT / "data" / "embeddings" / "prepared_data_v4_saprot.pt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "ogt_labels_bacdive_corrected.pt"

SANITY_GATE = 20.0  # °C — flag if |original - BacDive| > this


def parse_temp_val(t_str):
    if not t_str:
        return None
    t_str = str(t_str).strip()
    if "-" in t_str:
        parts = t_str.split("-")
        try:
            return sum(float(p) for p in parts) / len(parts)
        except ValueError:
            pass
    try:
        return float(t_str)
    except ValueError:
        return None


def query_single_taxid(tax_id, name):
    """Worker function to query a single NCBI tax_id using its scientific name."""
    if not name:
        return tax_id, None
    import bacdive
    client = bacdive.BacdiveClient()
    try:
        result_count = client.search(taxonomy=name)
        if result_count > 0:
            # Optimize: only retrieve first 5 strains to avoid huge page loads
            client.result["results"] = client.result["results"][:5]
            client.result["next"] = None
            
            temps = []
            for strain in client.retrieve():
                # Check both CamelCase/spaced keys and lowercase/underscored keys
                culture = strain.get("Culture and growth conditions", strain.get("culture_growth_condition", {}))
                temp_list = culture.get("culture temp", culture.get("culture_temp", []))
                
                for temp_entry in temp_list:
                    t = temp_entry.get("temperature", temp_entry.get("temp"))
                    test_type = temp_entry.get("type", "").lower()
                    t_val = parse_temp_val(t)
                    if t_val is not None and ("growth" in test_type or "optimum" in test_type or test_type == ""):
                        temps.append(t_val)
            if temps:
                return tax_id, float(np.median(temps))
            return tax_id, None
        return tax_id, None
    except Exception as e:
        return tax_id, "error"


def query_bacdive_for_taxids(unique_taxids: list, cache: dict, lineage_map: dict) -> dict:
    """Query BacDive API for growth temperatures by organism scientific name in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    taxid_to_ogt = {}
    to_query = [t for t in unique_taxids if t not in cache]
    
    # Fill cache values into taxid_to_ogt first
    for tax_id in unique_taxids:
        if tax_id in cache:
            taxid_to_ogt[tax_id] = cache[tax_id]

    if not to_query:
        return taxid_to_ogt

    print(f"  Starting parallel query for {len(to_query)} taxids with 16 workers...")
    
    found = sum(1 for v in cache.values() if v is not None)
    errors = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {}
        for t in to_query:
            # Map taxid to scientific name
            meta = lineage_map.get(str(t), {})
            name = meta.get("name")
            futures[executor.submit(query_single_taxid, t, name)] = t
        
        for future in as_completed(futures):
            tax_id = futures[future]
            try:
                _, result = future.result()
            except Exception as e:
                result = "error"
            
            completed += 1
            if result == "error":
                errors += 1
                cache[tax_id] = None
            else:
                cache[tax_id] = result
                if result is not None:
                    taxid_to_ogt[tax_id] = result
                    found += 1
            
            # Progress & incremental cache save
            if completed % 50 == 0 or completed == len(to_query):
                print(f"  Progress: {completed}/{len(to_query)} | Found: {found} | Errors: {errors}")
                with open(CACHE_FILE, 'w') as f:
                    json.dump(cache, f)

    return taxid_to_ogt


def apply_bacdive_labels(data: dict, taxid_to_ogt: dict, dry_run: bool = False) -> dict:
    """Replace OGT labels with BacDive values where available, with sanity gate."""
    tax_ids = data['train_ogt']['tax_id']
    original_labels = data['train_ogt']['ogt_consensus'].clone()
    new_labels = original_labels.clone()

    stats = {
        'total': len(tax_ids),
        'replaced': 0,
        'flagged_large_diff': 0,
        'no_bacdive_match': 0,
        'bacdive_null': 0,
    }
    flagged_examples = []

    for i, tax_id in enumerate(tax_ids):
        # Convert tax_id to string for cache lookup
        t_key = str(tax_id)
        if t_key in taxid_to_ogt and taxid_to_ogt[t_key] is not None:
            bacdive_ogt = taxid_to_ogt[t_key]
            original_ogt = float(original_labels[i])
            diff = abs(original_ogt - bacdive_ogt)

            if diff < SANITY_GATE:
                new_labels[i] = bacdive_ogt
                stats['replaced'] += 1
            else:
                stats['flagged_large_diff'] += 1
                if len(flagged_examples) < 20:
                    flagged_examples.append({
                        'tax_id': tax_id,
                        'original': original_ogt,
                        'bacdive': bacdive_ogt,
                        'diff': diff
                    })
        elif t_key in taxid_to_ogt:
            stats['bacdive_null'] += 1
        else:
            stats['no_bacdive_match'] += 1

    # Compute label change statistics
    diffs = (new_labels - original_labels).abs()
    changed_mask = diffs > 0.01

    print("\n=== Phase 0: BacDive OGT Replacement Statistics ===")
    print(f"  Total sequences:         {stats['total']:>10,}")
    print(f"  Labels replaced:         {stats['replaced']:>10,} ({stats['replaced']/stats['total']*100:.1f}%)")
    print(f"  Flagged (>20°C diff):    {stats['flagged_large_diff']:>10,}")
    print(f"  No BacDive match:        {stats['no_bacdive_match']:>10,}")
    print(f"  BacDive found, no temp:  {stats['bacdive_null']:>10,}")
    if stats['replaced'] > 0:
        print(f"\n  Mean label change (replaced): {diffs[changed_mask].mean():.2f}°C")
        print(f"  Max label change:             {diffs.max():.2f}°C")

    if flagged_examples:
        print(f"\n  Flagged examples (>20°C disagreement):")
        for ex in flagged_examples[:10]:
            print(f"    tax_id={ex['tax_id']}: original={ex['original']:.1f} → BacDive={ex['bacdive']:.1f} (Δ={ex['diff']:.1f}°C)")

    if not dry_run:
        data['train_ogt']['ogt_consensus'] = new_labels
        data['train_ogt']['ogt_bacdive_corrected'] = True

    return data, stats


def main():
    parser = argparse.ArgumentParser(description="Phase 0: BacDive OGT label replacement")
    parser.add_argument("--cache-only", action="store_true", help="Only query BacDive and build cache, don't modify data")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without modifying data")
    args = parser.parse_args()

    print("Phase 0: BacDive OGT Label Replacement")
    print(f"  Data file: {DATA_FILE}")
    print(f"  Sanity gate: {SANITY_GATE}°C")

    # Load lineages mapping
    lineage_path = PROJECT_ROOT / "data" / "cleaner_data" / "taxid_lineages.json"
    print(f"Loading lineages from {lineage_path}...")
    with open(lineage_path) as f:
        lineage_map = json.load(f)

    # Load data
    print("\nLoading data...")
    data = torch.load(DATA_FILE, map_location='cpu', weights_only=False)
    tax_ids = data['train_ogt']['tax_id']
    unique_taxids = sorted(set(tax_ids))
    print(f"  {len(tax_ids):,} sequences, {len(unique_taxids):,} unique organisms")

    # Load or create cache
    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        # Ensure all keys in cache are strings
        cache = {str(k): v for k, v in cache.items()}
        print(f"  Loaded cache: {len(cache)} entries ({sum(1 for v in cache.values() if v is not None)} with temps)")

    # Query BacDive
    uncached = [t for t in unique_taxids if str(t) not in cache]
    if uncached:
        print(f"\n  Querying BacDive for {len(uncached)} uncached organisms...")
        taxid_to_ogt = query_bacdive_for_taxids(uncached, cache, lineage_map)
    else:
        print("  All organisms already cached.")
        taxid_to_ogt = {k: v for k, v in cache.items() if v is not None}

    # Save final cache
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)
    print(f"  Cache saved: {len(cache)} entries")

    coverage = sum(1 for t in unique_taxids if cache.get(str(t)) is not None)
    print(f"  BacDive coverage: {coverage}/{len(unique_taxids)} organisms ({coverage/len(unique_taxids)*100:.1f}%)")

    if args.cache_only:
        print("\n  --cache-only: Stopping after BacDive query.")
        return

    # Apply labels
    print("\nApplying BacDive labels...")
    data, stats = apply_bacdive_labels(data, {str(k): v for k, v in cache.items() if v is not None}, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\nSaving corrected data to {OUTPUT_FILE}...")
        torch.save(data, OUTPUT_FILE)
        print("  Done.")
    else:
        print("\n  --dry-run: No data modified.")


if __name__ == "__main__":
    main()
