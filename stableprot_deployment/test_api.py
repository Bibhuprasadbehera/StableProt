#!/usr/bin/env python3
"""Simple diagnostic script to test the running StableProt server."""

import sys
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    print("[1/4] Testing /health ...")
    r = requests.get(f"{BASE_URL}/health")
    print(f"Status {r.status_code}: {r.json()}")
    assert r.status_code == 200

def test_predict():
    print("\n[2/4] Testing /api/predict ...")
    seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    payload = {"sequence": seq}
    r = requests.post(f"{BASE_URL}/api/predict", json=payload)
    print(f"Status {r.status_code}:")
    data = r.json()
    print(f"  Tm: {data.get('tm_pred')} ± {data.get('tm_conf')} °C ({data.get('thermal_tier')})")
    print(f"  OGT: {data.get('ogt_pred')} ± {data.get('ogt_conf')} °C")
    assert r.status_code == 200

def test_identify_loops():
    print("\n[3/4] Testing /api/identify_loops ...")
    seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    r = requests.post(f"{BASE_URL}/api/identify_loops", json={"sequence": seq})
    data = r.json()
    print(f"Status {r.status_code}: Found {len(data.get('loops', []))} loop regions.")
    assert r.status_code == 200

def test_evaluate_design():
    print("\n[4/4] Testing /api/evaluate_design ...")
    wt = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    mut = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGP"
    r = requests.post(f"{BASE_URL}/api/evaluate_design", json={"wt_sequence": wt, "mutant_sequence": mut})
    data = r.json()
    print(f"Status {r.status_code}: Delta Tm = {data.get('delta_tm')} °C ({data.get('thermal_effect')})")
    assert r.status_code == 200

if __name__ == "__main__":
    try:
        test_health()
        test_predict()
        test_identify_loops()
        test_evaluate_design()
        print("\nAll tests passed successfully!")
    except requests.exceptions.ConnectionError:
        print(f"\nCould not connect to {BASE_URL}. Make sure the server is running.")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
