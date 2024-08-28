import os
import requests
import tarfile
from tqdm import tqdm

def download_file(url, dest_folder):
    """Download a file from a URL and save it to the destination folder."""
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    filename = os.path.join(dest_folder, url.split('/')[-1])
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))
    
    print(f"Downloaded {filename}")
    return filename

def extract_tgz(tgz_path, extract_to):
    """Extract a .tgz file to a specified directory."""
    with tarfile.open(tgz_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_to)
    print(f"Extracted {tgz_path} to {extract_to}")

def remove_file(file_path):
    """Remove a file."""
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Removed {file_path}")

def download_imagenette(dest_folder):
    """Download and extract the full-size Imagenette dataset."""
    imagenette_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    tgz_file_path = download_file(imagenette_url, dest_folder)
    
    # Extract the tgz file
    extract_tgz(tgz_file_path, dest_folder)
    
    # Remove the tgz file
    remove_file(tgz_file_path)

def download_compas(dest_folder):
    """Download COMPAS dataset directly from the provided link."""
    dataset_url = "https://bwsyncandshare.kit.edu/s/F7acYAponsKxe26/download/compas-scores-two-years.csv"
    download_file(dataset_url, dest_folder)
    print(f"Downloaded COMPAS dataset to {dest_folder}")

if __name__ == "__main__":
    data_dir = "data"
    
    # Download datasets
    print("Downloading Imagenette...")
    download_imagenette(data_dir)
    
    print("Downloading COMPAS...")
    compas_dir = "data/COMPAS"
    download_compas(compas_dir)
    
    print("All downloads completed.")
