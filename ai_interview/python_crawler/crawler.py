import os
from typing import List, TypedDict
from argparse import Namespace

import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from config import header_config

from argparse import ArgumentParser
from dotenv import load_dotenv

load_dotenv('.env.local')


def reader_url(url: str):
    # 解析 url 中的域名
    request_url = f"https://r.jina.ai/{url}"
    domain = url.split('/')[2]
    headers = {
        "Authorization": "Bearer " + os.getenv("API_KEY", ""),
        **header_config.get(domain, {}),
    }
    response = requests.get(request_url, headers=headers)
    return response.text


def save_to_file(text: str, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)


def get_name_from_url(url: str):
    return url.split('/')[-1].split('?')[0].split('.')[0]


def create_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_cache(dir_path: str):
    cache_files: List[str] = []
    for file in os.listdir(dir_path):
        if file.endswith('.md'):
            # 去掉文件名中的 .md 后缀
            cache_files.append(file.replace('.md', ''))
    return cache_files


def process_url(url, output_dir: str):
    text = reader_url(url)
    error_list = ['<!DOCTYPE html>',
                  'AssertionFailureError', 'ServiceCrashedError', 'AuthenticationFailedError']
    if any(error in text for error in error_list):
        return
    save_to_file(text, f"{output_dir}/{get_name_from_url(url)}.md")


def get_urls_from_file(file_path: str, caches: List[str]) -> List[str]:
    urls = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            url = line.strip()
            if url and get_name_from_url(url) not in caches:
                urls.add(url)
    return list(urls)


class CrawlerArgs(TypedDict):
    output_dir: str
    urls: str


def main(args: CrawlerArgs):
    create_dir(args['output_dir'])
    cache = load_cache(args['output_dir'])
    valid_urls = get_urls_from_file(args['urls'], cache)

    # 设置线程池，max_workers 可以根据需要调整
    with ThreadPoolExecutor(max_workers=3) as executor:
        # 使用 list() 确保所有任务完成
        list(tqdm(
            executor.map(lambda url: process_url(
                url, args['output_dir']), valid_urls),
            total=len(valid_urls),
            desc="Processing URLs"
        ))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='输出目录路径', required=True)
    parser.add_argument('--urls', type=str, help='urls 文件路径', required=True)
    args = parser.parse_args()

    args_dict: CrawlerArgs = {"output_dir": args.output_dir, "urls": args.urls}
    main(args_dict)
