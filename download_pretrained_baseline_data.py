import os
import requests
import zipfile

def download_and_unzip(url, destination_path):
    """
    Downloads a ZIP file from the given URL, extracts its contents to the specified destination path, 
    and deletes the ZIP file after extraction.

    :param url: URL of the ZIP file to download
    :param destination_path: Directory where the extracted files should be saved
    """
    # Ensure the destination directory exists
    os.makedirs(destination_path, exist_ok=True)

    # Extract the filename from the URL
    zip_filename = os.path.join(destination_path, url.split('/')[-1])

    # Download the ZIP file
    print(f"Downloading {zip_filename} from {url}...")
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Save the downloaded content to a file
    with open(zip_filename, 'wb') as file:
        file.write(response.content)

    print(f"Downloaded {zip_filename}")

    # Extract the ZIP file
    print(f"Extracting {zip_filename} to {destination_path}...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(destination_path)

    print(f"Extraction complete.")

    # Delete the ZIP file
    os.remove(zip_filename)
    print(f"Deleted the ZIP file {zip_filename}")

if __name__ == "__main__":
    data_dir = "."
    # 
    download_and_unzip(
        url='https://bwsyncandshare.kit.edu/s/Yg8b822zrRbqXXA/download/base_models2.zip',
        destination_path=data_dir
    )
