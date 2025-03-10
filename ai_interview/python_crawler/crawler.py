import os
from typing import List

import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from config import header_config
from urls import urls


def reader_url(url: str):
    # 解析 url 中的域名
    request_url = f"https://r.jina.ai/{url}"
    domain = url.split('/')[2]
    headers = {
        "Authorization": "Bearer jina_7ab526b26e93458ebaeeaa9efeba851dKLJvSVMIHMTDpPH8CogEmDhLZ5-K",
        **header_config.get(domain, {}),
    }

    response = requests.get(request_url, headers=headers)
    return response.text


def save_to_file(text: str, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)


def get_name_from_url(url: str):
    return url.split('/')[-1].split('?')[0]


def load_cache(dir_path: str):
    cache_files: List[str] = []
    for file in os.listdir(dir_path):
        if file.endswith('.md'):
            # 去掉文件名中的 .md 后缀
            cache_files.append(file.replace('.md', ''))
    return cache_files


def process_url(url):
    text = reader_url(url)
    if '<!DOCTYPE html>' in text or 'AssertionFailureError' in text or 'ServiceCrashedError' in text:
        return
    save_to_file(text, f"raw_data/{get_name_from_url(url)}.md")


def main():
    cache = load_cache('raw_data')
    valid_urls = [
        url for url in set(urls)
        if 'nowcoder' in url and get_name_from_url(url) not in cache
    ]

    # 设置线程池，max_workers 可以根据需要调整
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 使用 list() 确保所有任务完成
        list(tqdm(
            executor.map(process_url, valid_urls),
            total=len(valid_urls),
            desc="Processing URLs"
        ))


if __name__ == "__main__":
    main()
