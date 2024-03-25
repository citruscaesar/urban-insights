from shutil import rmtree
from pathlib import Path
from threading import local

def reset_dir(*args) -> Path:
    path = Path(*args)
    if path.is_dir() and path.exists():
        rmtree(path.as_posix())
    path.mkdir(parents = True)
    return path

def validate_dir(*args) -> Path:
    path = Path(*args)
    path.mkdir(exist_ok=True, parents=True)
    return path

def is_valid_remote(remote_url: str) -> bool:
    assert isinstance(remote_url, str), "URL must be of string type"

    # TODO : Find a way to validate remote URLS, boto3 for s3?

    if "//" in remote_url:
        return True
    return False

def is_valid_path(local_path: str | Path) -> bool:
    local_path = Path(local_path)
    if local_path.exists():
        return True
    return False

def get_local_path_from_remote(remote_url: str) -> Path:
    assert isinstance(remote_url, str), "URL must be of string type"
    return Path(Path.home(), *remote_url.split('//')[-1].split('/'))

def is_empty(local_path: str|Path) -> bool:
    local_path = Path(local_path)
    return not list(local_path.iterdir())