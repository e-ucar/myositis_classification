import requests
from zipfile import ZipFile
import os

IMG_DIRECTORY = "./"
IMG_GD_ID = "1z5XCOOi0-Yz5UTbmKm6egRgW5ukVrUvf"
IMG_ZIP_FILE = "./images.zip"

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_google_drive_file(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def download_images():
    if not os.path.exists(IMG_DIRECTORY):
        os.makedirs(IMG_DIRECTORY)
        print("Downloading images to " + IMG_ZIP_FILE)
        download_google_drive_file(IMG_GD_ID, IMG_ZIP_FILE)
        print("Finished downloading images.")
        print("Unzipping " + IMG_ZIP_FILE)
        with ZipFile(IMG_ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(IMG_DIRECTORY)
        print("Finished unzipping images.")

if __name__ == '__main__':
    download_images()