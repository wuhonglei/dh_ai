import os

from tqdm import tqdm

from train import main
from utils.category import load_category_list, get_category_by_id, get_leaf_level_category_list

if __name__ == '__main__':
    base_dir = './dataset/leaf_levels/'
    category_list = load_category_list('./json/mtsku_category_tree.json')
    category_progress = tqdm(os.listdir(base_dir))
    for level1_category_id in category_progress:
        category_progress.set_description(f'Processing {level1_category_id}')
        data_dir = f'{base_dir}/{level1_category_id}'
        category = get_category_by_id(level1_category_id, category_list)
        if not category:
            continue

        leaf_category_list = get_leaf_level_category_list(category)
        category_id_list = [str(item['id']) for item in leaf_category_list]
        main(data_dir=data_dir, category_id_list=category_id_list,
             model_name=f'leaf_levels_{level1_category_id}', tags=['leaf_levels', level1_category_id])
