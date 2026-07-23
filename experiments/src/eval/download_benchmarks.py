import os
import requests
import tarfile
import subprocess

def download_file_from_google_drive(file_id, destination):
    print(f"Downloading Google Drive file ID {file_id} to {destination}...")
    URL = "https://drive.usercontent.google.com/download"
    session = requests.Session()
    params = {'id': file_id, 'export': 'download', 'confirm': 't'}
    response = session.get(URL, params=params, stream=True)
    if response.status_code == 200:
        save_response_content(response, destination)
        print(f"Finished downloading {destination}.")
    else:
        print(f"FAILED to download from Google Drive. Status: {response.status_code}")

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def download_url(url, destination):
    print(f"Downloading {url} to {destination}...")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(destination, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        print(f"Finished downloading {destination}.")
    else:
        print(f"FAILED to download {url}. Status code: {r.status_code}")

def main():
    os.makedirs("data/emergent_benchmarks", exist_ok=True)
    
    # 1. Google Drive files (SaProt downstream tasks)
    gdrive_files = {
        "EC.tar.gz": "1VFLFA-jK1tkTZBVbMw8YSsjZqAqlVQVQ",
        "DeepLoc.tar.gz": "1dGlojkCt1DwUXWiUk4kXRGRNu5sz2uxf",
        "HumanPPI.tar.gz": "1ahgj-IQTtv3Ib5iaiXO_ASh2hskEsvoX",
        "Thermostability.tar.gz": "1I9GR1stFDHc8W3FCsiykyrkNprDyUzSz"
    }
    
    # Clean up bad downloads first if they are not real gzip files
    for filename in gdrive_files.keys():
        dest = os.path.join("data/emergent_benchmarks", filename)
        if os.path.exists(dest):
            # Check if it starts with '<!' (HTML) or is too small
            with open(dest, 'rb') as f:
                header = f.read(2)
            if header == b'<!' or os.path.getsize(dest) < 10000:
                print(f"Deleting bad file: {dest}")
                os.remove(dest)
    
    for filename, file_id in gdrive_files.items():
        dest = os.path.join("data/emergent_benchmarks", filename)
        if not os.path.exists(dest):
            download_file_from_google_drive(file_id, dest)
        else:
            print(f"{filename} already exists with correct size, skipping download.")
            
        # Extract the tar.gz files
        extract_dir = os.path.join("data/emergent_benchmarks", filename.replace(".tar.gz", ""))
        if not os.path.exists(extract_dir):
            print(f"Extracting {dest} to data/emergent_benchmarks...")
            try:
                with tarfile.open(dest, "r:gz") as tar:
                    tar.extractall(path="data/emergent_benchmarks")
                print(f"Extraction of {filename} complete.")
            except Exception as e:
                print(f"Failed to extract {filename}: {e}")
        else:
            print(f"Directory {extract_dir} already exists, skipping extraction.")

    # 2. eSOL dataset from HuggingFace
    esol_dir = "data/emergent_benchmarks/eSOL"
    os.makedirs(esol_dir, exist_ok=True)
    esol_urls = {
        "train.csv": "https://huggingface.co/datasets/AI4Protein/eSOL/resolve/main/train.csv",
        "test.csv": "https://huggingface.co/datasets/AI4Protein/eSOL/resolve/main/test.csv",
        "valid.csv": "https://huggingface.co/datasets/AI4Protein/eSOL/resolve/main/valid.csv"
    }
    for filename, url in esol_urls.items():
        dest = os.path.join(esol_dir, filename)
        if not os.path.exists(dest):
            download_url(url, dest)
        else:
            print(f"eSOL {filename} already exists, skipping.")

    # 3. CB513 dataset from HuggingFace
    cb513_dir = "data/emergent_benchmarks/CB513"
    os.makedirs(cb513_dir, exist_ok=True)
    cb513_url = "https://huggingface.co/datasets/proteinea/secondary_structure_prediction/resolve/main/CB513.csv"
    dest_cb513 = os.path.join(cb513_dir, "CB513.csv")
    if not os.path.exists(dest_cb513):
        download_url(cb513_url, dest_cb513)
    else:
        print("CB513 already exists, skipping.")

    # 4. Remote Homology / Fold Prediction dataset from HuggingFace
    scop_dir = "data/emergent_benchmarks/scop"
    os.makedirs(scop_dir, exist_ok=True)
    scop_urls = {
        "train.parquet": "https://huggingface.co/datasets/proteinglm/fold_prediction/resolve/main/data/train-00000-of-00001.parquet",
        "test.parquet": "https://huggingface.co/datasets/proteinglm/fold_prediction/resolve/main/data/test-00000-of-00001.parquet",
        "valid.parquet": "https://huggingface.co/datasets/proteinglm/fold_prediction/resolve/main/data/valid-00000-of-00001.parquet"
    }
    for filename, url in scop_urls.items():
        dest = os.path.join(scop_dir, filename)
        if not os.path.exists(dest):
            download_url(url, dest)
        else:
            print(f"SCOP {filename} already exists, skipping.")

    # 5. LiveProteinBench from GitHub
    lpb_dir = "data/emergent_benchmarks/LiveProteinBench"
    if not os.path.exists(lpb_dir):
        print("Cloning LiveProteinBench from GitHub...")
        subprocess.run(["git", "clone", "https://github.com/Rongdingyi/LiveProteinBench.git", lpb_dir], check=True)
        print("Finished cloning LiveProteinBench.")
    else:
        print("LiveProteinBench already cloned, skipping.")

if __name__ == "__main__":
    main()
