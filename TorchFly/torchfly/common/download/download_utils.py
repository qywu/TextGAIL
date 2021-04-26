import requests
import tqdm
import gdown
import os


def http_get(url, filename, proxies=None):
    req = requests.get(url, stream=True, proxies=proxies)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm.tqdm(unit="B", total=total)

    with open(filename, "wb") as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                f.write(chunk)

    progress.close()


def http_download(url, folder, filename):
    cache_dir = os.path.join(os.getenv("HOME"), ".cache", "torchfly", folder)
    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, filename)

    if not os.path.exists(filepath):
        http_get(url, filepath)
    else:
        print(f"{filename} exists!")

    return filepath


def gdrive_download(url, folder, filename):
    cache_dir = os.path.join(os.getenv("HOME"), ".cache", "torchfly", folder)
    os.makedirs(cache_dir, exist_ok=True)

    filepath = os.path.join(cache_dir, filename)
    gdown.cached_download(url, filepath, quiet=False)

    return filepath