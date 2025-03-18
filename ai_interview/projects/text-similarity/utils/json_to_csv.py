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
    for i in tqdm(range(len(dataset))):
        title, content = dataset[i]
        processed_data.append([i, title, content])

    # 写入CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'title', 'content'])
        writer.writerows(processed_data)


if __name__ == "__main__":
    start_time = time.time()
    convert_to_csv(NewsDatasetJson("../data/origin/sohu_data.json"),
                   "../data/origin/sohu_data.csv")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
