from dataset import NewsDataset
import csv
import time
from tqdm import tqdm


def transform_json_to_csv(json_path: str, csv_path: str):
    dataset = NewsDataset(json_path)
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['title', 'content'])
        for i in tqdm(range(len(dataset))):
            item = dataset[i]
            writer.writerow([item['title'], item['content']])


if __name__ == "__main__":
    start_time = time.time()
    transform_json_to_csv("sohu_data.json", "sohu_data.csv")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
