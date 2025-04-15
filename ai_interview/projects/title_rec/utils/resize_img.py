import os
from tqdm import tqdm
import pandas as pd
import albumentations as A
import cv2
import time
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from typing import List, Tuple
from config import clean_dataset_dir, src_img_dir, resize_img_dir

pipeline = A.Compose([
    A.Resize(224, 224, interpolation=cv2.INTER_LINEAR),
])


def resize_single_img(img_path: str, save_path: str, pipeline: A.Compose):
    img = cv2.imread(img_path)
    img = pipeline(image=img)['image']
    cv2.imwrite(save_path, img)


def process_imgs(src_imgs: list[str], save_dir: str, pipeline: A.Compose):
    for img_path in tqdm(src_imgs, desc='Processing images'):
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        resize_single_img(img_path, save_path, pipeline)


def get_src_imgs(csv_path: str, src_img_dir: str) -> List[str]:
    """Get all image paths from source directory."""
    df = pd.read_csv(csv_path)
    success_df = df[df['download_success'] == 1]
    src_imgs = [os.path.join(src_img_dir, f)
                for f in success_df['main_image_name'].tolist()]
    return src_imgs


def empty_dir(dir_path: str):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def process_single_image(args: Tuple[str, str, A.Compose]) -> None:
    """Process a single image with the given pipeline."""
    img_path, save_path, pipeline = args
    img = cv2.imread(img_path)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = pipeline(image=img)
    transformed_img = transformed['image']
    transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, transformed_img)


def process_imgs_parallel(src_imgs: List[str], save_dir: str, pipeline: A.Compose, use_parallel: bool) -> None:
    """Process images in parallel using threading."""
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    if not use_parallel:
        for img_path in tqdm(src_imgs, desc="Processing images sequentially"):
            process_single_image((img_path, os.path.join(
                save_dir, os.path.basename(img_path)), pipeline))
        return

    # Prepare arguments for each image
    args_list = []
    for img_path in src_imgs:
        filename = os.path.basename(img_path)
        save_path = os.path.join(save_dir, filename)
        args_list.append((img_path, save_path, pipeline))

    # Get number of CPU cores and set thread count
    # For I/O bound tasks like image processing, we can use more threads than CPU cores
    num_threads = min(multiprocessing.cpu_count() * 2, len(args_list))

    # Process images in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(
            executor.map(process_single_image, args_list),
            total=len(args_list),
            desc="Processing images in parallel"
        ))


def main():
    csv_names = ['valid.csv', 'test.csv']
    save_dir = resize_img_dir
    use_parallel = True
    empty_dir(save_dir)
    time_start = time.time()
    src_imgs = []
    for csv_name in csv_names:
        src_imgs.extend(get_src_imgs(os.path.join(
            clean_dataset_dir, csv_name), src_img_dir))
    process_imgs_parallel(src_imgs, save_dir, pipeline, use_parallel)
    time_end = time.time()
    print(f'time cost: {time_end - time_start}s')


if __name__ == '__main__':
    main()
