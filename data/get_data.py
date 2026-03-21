import os
import zipfile
import urllib.request

DATA_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "dataset.zip")
EXTRACTED_FOLDER = os.path.join(DATA_DIR, "raw")

EXPECTED_FILE = "household_power_consumption.txt"
EXPECTED_PATH = os.path.join(EXTRACTED_FOLDER, EXPECTED_FILE)

def is_zip_valid(zip_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            bad_file = z.testzip()
            return bad_file is None
    except zipfile.BadZipFile:
        return False

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(ZIP_PATH):
        print("Zip file already exists. Checking integrity...")
        if is_zip_valid(ZIP_PATH):
            print("Zip file is valid. Skipping download.")
            return
        else:
            print("Zip file is corrupted. Deleting and re-downloading...")
            os.remove(ZIP_PATH)

    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    print("Download complete.")

def extract_dataset():
    os.makedirs(EXTRACTED_FOLDER, exist_ok=True)

    if os.path.exists(EXPECTED_PATH):
        print(f"{EXPECTED_FILE} already exists. Skipping extraction.")
        return

    if not is_zip_valid(ZIP_PATH):
        raise RuntimeError("Zip file is corrupted. Please re-download.")

    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACTED_FOLDER)
    print("Extraction complete.")

if __name__ == "__main__":
    download_dataset()
    extract_dataset()