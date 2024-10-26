"""
将其他地区的  category 映射为 sg 地区的 category
"""

import pandas as pd

from config import countries
category_map = {
    'sg': {
        # values 为 list 时，只要满足其中一个即可
        'Home & Living': [{
            'type': 'common',  # 各地区通用
            'columns': {
                # 该列的值存在于 list 中
                'imp_level1_category_1d': ['Home & Living'],
                'pv_level1_category_1d': ['Home & Living'],
                'order_level1_category_1d': ['Home & Living'],
            },
            'min_count': 3,  # 最小出现次数
        }],
        'Food & Beverages': [{
            'type': 'common',
            'columns': {
                'imp_level1_category_1d': ['Food & Beverages'],
                'pv_level1_category_1d': ['Food & Beverages'],
                'order_level1_category_1d': ['Food & Beverages'],
            },
            'min_count': 3,
        }],
        'Home Appliances': [{
            'type': 'common',
            'columns': {
                'imp_level1_category_1d': ['Home Appliances'],
                'pv_level1_category_1d': ['Home Appliances'],
                'order_level1_category_1d': ['Home Appliances'],
            },
            'min_count': 3,
        }],
        'Mobile & Gadgets': [{
            'type': 'common',
            'columns': {
                'imp_level1_category_1d': ['Mobile & Gadgets'],
                'pv_level1_category_1d': ['Mobile & Gadgets'],
                'order_level1_category_1d': ['Mobile & Gadgets'],
            },
            'min_count': 3,
        },
            {
            'type': 'common',
            'columns': {
                'imp_level1_category_1d': ['audio'],
                'pv_level1_category_1d': ['audio'],
                'order_level1_category_1d': ['audio'],
            },
            'min_count': 3,
        }],
        'Sports & Outdoors': [{
            'type': 'common',
            'columns': {
                'imp_level1_category_1d': ['Sports & Outdoors'],
                'pv_level1_category_1d': ['Sports & Outdoors'],
                'order_level1_category_1d': ['Sports & Outdoors'],
            },
            'min_count': 3,
        }],
        'Computers & Peripherals': [{
            'type': 'common',
            'columns': {
                'imp_level1_category_1d': ['Computers & Accessories'],
                'pv_level1_category_1d': ['Computers & Accessories'],
                'order_level1_category_1d': ['Computers & Accessories'],
            },
            'min_count': 3,
        }],
        'Beauty & Personal Care': [{
            'type': 'common',
            'columns': {
                'imp_level1_category_1d': ['Beauty'],
                'pv_level1_category_1d': ['Beauty'],
                'order_level1_category_1d': ['Beauty'],
            },
            'min_count': 3,
        }],
        'Health & Wellness': [{
            'type': 'common',
            'columns': {
                'imp_level1_category_1d': ['Health'],
                'pv_level1_category_1d': ['Health'],
                'order_level1_category_1d': ['Health'],
            },
            'min_count': 3,
        }],
        'Toys, Kids & Babies': [{
            'type': 'common',
            'columns': {
                'imp_level1_category_1d': ['Mom & Baby'],
                'pv_level1_category_1d': ['Mom & Baby'],
                'order_level1_category_1d': ['Mom & Baby'],
            },
            'min_count': 3,
        }],
        "Women's Apparel": [{
            'type': 'common',
            'columns': {
                'imp_level1_category_1d': ['Women Clothes'],
                'pv_level1_category_1d': ['Women Clothes'],
                'order_level1_category_1d': ['Women Clothes'],
            },
            'min_count': 3,
        }],
        'Hobbies & Books': [{
            'type': 'common',
            'columns': {
                'imp_level1_category_1d': ['Hobbies & Collections'],
                'pv_level1_category_1d': ['Hobbies & Collections'],
                'order_level1_category_1d': ['Hobbies & Collections'],
            },
            'min_count': 3,
        }],
        'Jewellery & Accessories': [{
            'type': 'common',
            'columns': {
                'imp_level1_category_1d': ['Fashion Accessories'],
                'pv_level1_category_1d': ['Fashion Accessories'],
                'order_level1_category_1d': ['Fashion Accessories'],
            },
            'min_count': 3,
        }],
        "Women's Bags": [{
            'type': 'common',
            'columns': {
                'imp_level1_category_1d': ['Women Bags'],
                'pv_level1_category_1d': ['Women Bags'],
                'order_level1_category_1d': ['Women Bags'],
            },
            'min_count': 3,
        }],
        "Men's Wear": [{
            'type': 'common',
            'columns': {
                'imp_level1_category_1d': ['Men Clothes'],
                'pv_level1_category_1d': ['Men Clothes'],
                'order_level1_category_1d': ['Men Clothes'],
            },
            'min_count': 3,
        }],
        "Women's Shoes": [{
            'type': 'common',
            'columns': {
                'imp_level1_category_1d': ['Women Shoes'],
                'pv_level1_category_1d': ['Women Shoes'],
                'order_level1_category_1d': ['Women Shoes'],
            },
            'min_count': 3,
        }],
        'Automotive': [{
            'type': 'common',
            'columns': {
                'imp_level1_category_1d': ['Automobiles'],
                'pv_level1_category_1d': ['Automobiles'],
                'order_level1_category_1d': ['Automobiles'],
            },
            'min_count': 3,
        }],
        'Dining, Travel & Services': [{
            'type': 'common',
            'columns': {
                'imp_level1_category_1d': ['Tickets, Vouchers & Services'],
                'pv_level1_category_1d': ['Tickets, Vouchers & Services'],
                'order_level1_category_1d': ['Tickets, Vouchers & Services'],
            },
            'min_count': 3,
        }],
        'Platform Terms (Lazada / Shopee)': [{
            'type': 'country',  # 该类别每个国家判断条件不同
            'my': {
                'columns': {
                    'Category': ['Platform Terms (Lazada/Shopee)'],
                },
                'min_count': 1,
            },
            'th': {
                'columns': {
                    'Category': ['Platform Terms (Lazada/Shopee)'],
                },
                'min_count': 1,
            },
            'tw': {
                'columns': {
                    'Category': ['Platform Terms ( Shopee / Momo)'],
                },
                'min_count': 1,
            },
            'id': {},  # 没有该类别
            'vn': {
                'columns': {
                    'Category': ['Platform Terms (Shopee/ Lazada)'],
                },
                'min_count': 1,
            },
            'ph': {
                'columns': {
                    'Category': ['Platform Keywords (Shopee/Lazada)'],
                },
                'min_count': 1,
            },
            'br': {
                'columns': {
                    'Category': ['Platform Terms (Shopee / Mercado)'],
                },
                'min_count': 1,
            },
            'mx': {
                'columns': {
                    'Category': ['Platform Terms (Mercado/Shopee)'],
                },
                'min_count': 1,
            },
            'co': {},  # 没有该类别
            'cl': {
                'columns': {
                    'Category': ['Platform Terms (Mercado/Shopee)'],
                },
                'min_count': 1,
            }
        }]
    }
}


target_country = 'sg'  # 翻译为英文

for country in countries:
    print(f'Processing {country}...')
    df = pd.read_csv(f'./data/translated_csv/{country}.csv')
    old_category_name = 'Category'
    new_category_name = f'{target_country}_category'
    if country == target_country:
        # 删除 Category 列值为 Miscellaneous (Adult) 或 Miscellaneous 的行
        df = df[~df[old_category_name].isin(
            ['Miscellaneous (Adult)', 'Miscellaneous'])]

        df[new_category_name] = df[old_category_name]
        df.to_csv(
            f'./data/after_category/{country}.csv', index=False)
        continue

    # 遍历 df 的每一行
    for i, row in df.iterrows():
        print(f'Processing {country} {i}...')
        category = row[old_category_name]
        for mapped_category_name, condition_list in category_map[target_country].items():
            for condition in condition_list:
                if condition['type'] == 'country':
                    condition = condition[country]
                    if not condition:
                        continue

                match_count = 0
                for columns_name, value_list in condition['columns'].items():
                    if df.at[i, columns_name] in value_list:
                        match_count += 1

                if match_count >= condition['min_count']:
                    df.at[i, new_category_name] = mapped_category_name
                    print(f'{country} {i} matched {mapped_category_name}')
                    break

    df.to_csv(
        f'./data/after_category/{country}.csv', index=False)
    print(f'Finished {country}')
