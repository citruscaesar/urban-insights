#General Purpose Imports
from pathlib import Path

# Asyc Download Imports 
import aiohttp
import aiofiles 
from tqdm.asyncio import tqdm_asyncio

class DatasetDownloader:
    def __init__(self, downloads:Path, urls:dict):

        # init required directories
        if not (downloads.exists() and downloads.is_dir()):
            downloads.mkdir(exist_ok=True, parents=True)
            print(f"download directory at {downloads}")
        self.download_dir = downloads

        # src_urls is a dictionary such that
        # src_urls[filename:str]  = url:str
        self.src_urls = urls
        
    async def async_download_one_file(self, session, url:str, file_path:Path):
        """Download one file from url and save to disk at file_path"""
        #TODO: How to use tqdm for each coroutine in the notebook
        async with session.get(url, ssl = False) as r:
            #total_size = int(r.headers.get('content-length', 0))
            async with aiofiles.open(file_path, "wb") as f:
                #progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading")
                async for chunk in r.content.iter_any():
                    await f.write(chunk)
                    #progress_bar.update(len(chunk))

    async def download_files(self, async_download_one_file = None) -> None:
        if not async_download_one_file:
            async_download_one_file = self.async_download_one_file
        #Download files from self.src_urls, skip if already_downloaded
        timeout = aiohttp.ClientTimeout(total = None)
        async with aiohttp.ClientSession(timeout=timeout, cookie_jar=aiohttp.CookieJar()) as session:
            coroutines = list()
            for file_name, url in self.src_urls.items():
                file_path = self.download_dir / file_name 
                coroutines.append(async_download_one_file(session, url, file_path))
            await tqdm_asyncio.gather(*coroutines)   
    
    def validate_download(self, downloaded_file_sizes: dict) -> None:
        #TODO: Implement Validate Downloads
        pass