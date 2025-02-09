from train import main
from utils.category import load_category_list

if __name__ == '__main__':
    category_list = load_category_list('./json/mtsku_category_tree.json')
    category_id_list = [str(item['id']) for item in category_list]
    main(data_dir='./dataset/level1/', category_id_list=category_id_list,
         model_name='level1', tags=['level1'])
