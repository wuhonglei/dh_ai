import os
import random
import time
from typing import List, TypedDict

import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from playwright.sync_api import sync_playwright  # type: ignore
from config import header_config

from argparse import ArgumentParser
from dotenv import load_dotenv

load_dotenv('.env.local')


def get_headers(url: str):
    domain = url.split('/')[2]
    return header_config.get(domain, {})


def reader_url(url: str, headers: dict):
    # 解析 url 中的域名
    request_url = f"https://r.jina.ai/{url}"
    headers = {
        "Authorization": "Bearer " + os.getenv("API_KEY", ""),
        **headers,
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


def open_url(url: str, headers: dict):
    """ 使用 playwright 打开 url 并返回 html """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(no_viewport=True)
        page.goto(url)
        # 随机等待 1-3 秒
        time.sleep(random.randint(1, 3))

        wait_for_selector = headers.get(
            'X-Wait-For-Selector', '').split(',')[0]

        if wait_for_selector:
            try:
                page.wait_for_selector(
                    wait_for_selector, timeout=15000)  # 等待 3 秒
            except Exception as e:
                print(e)
                return None

        selectors = headers.get('X-Target-Selector', '').split(',')
        content = {
            'title': page.title(),
            "url_source": url,
            "markdown_content": []
        }
        for selector in selectors:
            # 获取页面中所有符合 selector 的元素
            elements = page.query_selector_all(selector)
            for element in elements:
                content['markdown_content'].append(
                    element.text_content().strip())  # type: ignore

        markdown_content = f'Title: {content["title"].strip()}\n\n'
        markdown_content += f'URL Source: {content["url_source"].strip()}\n\n'
        markdown_content += '\n'.join(content['markdown_content'])
        return markdown_content


def process_url(url, output_dir: str):
    time.sleep(random.randint(1, 3))
    headers = get_headers(url)
    text = reader_url(url, headers)
    error_list = ['<!DOCTYPE html>',
                  'AssertionFailureError', 'ServiceCrashedError', 'AuthenticationFailedError']
    if any(error in text for error in error_list):
        text = open_url(url, headers)

    if text:
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
    with ThreadPoolExecutor(max_workers=1) as executor:
        # 使用 list() 确保所有任务完成
        list(tqdm(
            executor.map(lambda url: process_url(
                url, args['output_dir']), valid_urls),
            total=len(valid_urls),
            desc="Processing URLs"
        ))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        help='输出目录路径', default='./boss直聘')
    parser.add_argument('--urls', type=str,
                        help='urls 文件路径', default='./boss直聘.txt')
    args = parser.parse_args()

    args_dict: CrawlerArgs = {"output_dir": args.output_dir, "urls": args.urls}
    main(args_dict)
