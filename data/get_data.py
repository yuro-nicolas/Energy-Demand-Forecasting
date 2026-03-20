import os
import zipfile
import urllib.request

DATA_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "dataset.zip")
EXTRACTED_FOLDER = os.path.join(DATA_DIR, "raw")

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        print("Download complete.")
    else:
        print("Dataset already downloaded.")

def extract_dataset():
    os.makedirs(EXTRACTED_FOLDER, exist_ok=True)

    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACTED_FOLDER)
    print("Extraction complete.")

if __name__ == "__main__":
    download_dataset()
    extract_dataset()