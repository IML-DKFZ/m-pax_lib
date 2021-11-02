import os
import zipfile
import requests

def download_url(url, save_path):  # Chunk wise downloading to not overuse RAM
    r = requests.get(url, stream=True, allow_redirects=True)
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                f.flush()