from dataset import NewsDatasetJson
import csv
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def process_item(item):
    title, content = item
    return [title, content]


def convert_to_csv(dataset, output_file, max_workers=4):
    # 使用ThreadPoolExecutor处理数据
    processed_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用 list() 确保所有任务完成，tqdm 显示进度
        futures = list(tqdm(
            executor.map(process_item, dataset),
            total=len(dataset),
            desc="Processing data"
        ))
        processed_data.extend(futures)

    # 写入CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['title', 'content'])
        writer.writerows(processed_data)


if __name__ == "__main__":
    start_time = time.time()
    convert_to_csv(NewsDatasetJson("data/origin/sohu_data.json"),
                   "data/origin/sohu_data.csv")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
