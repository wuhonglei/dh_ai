"""
分析 csv 文件中的 Category 字段，统计每个国家的 Category 字段的分布情况
"""

import pandas as pd
import numpy as np

# from config import countries

countries = [
    'sg',
    'my',
    'th',
    'tw',
    'id',
    'vn',
    'ph',
    'br',
    'mx',
    'co',
    'cl'
]


category_name1 = 'imp_level1_category_1d'
category_name2 = 'pv_level1_category_1d'
category_name3 = 'order_level1_category_1d'

img_category_set = set([
    'Home & Living',
    'Health & Personal Care',
    'Hobbies & Stationery',
    'Groceries',
    'Sports & Travel',
    'Laptops & Computers',
    'Mobile Accessories',
    "Women's Apparel",
    'Motors',
    'Toys, Games & Collectibles',
    'Mobiles & Gadgets',
    'Home Appliances',
    "Men's Apparel",
    'Mobile & Gadgets',
    'Women Accessories',
    'Men Shoes',
    'Babies & Kids',
    'Makeup & Fragrances',
    'Pet Care',
    'Women Shoes',
    'Health',
    'Sports & Outdoors',
    '-',
    'Food & Beverages',
    'Home Entertainment',
    'Beauty',
    "Women's Bags",
    "Men's Bags & Accessories",
    'Women Clothes',
    'Cameras',
    'Men Clothes',
    'Hobbies & Collections',
    'Computers & Accessories',
    'Stationery',
    'Mom & Baby',
    'Gaming',
    'Books & Magazines',
    'Motorcycles',
    'Fashion Accessories',
    'Baby & Kids Fashion',
    'Pets',
    'Automobiles',
    'Audio',
    'Cameras & Drones',
    'Women Bags',
    'Watches',
    'Gaming & Consoles',
    'Digital Goods & Vouchers',
    'Travel & Luggage',
    'Men Bags',
    'Tickets, Vouchers & Services',
    'Muslim Fashion'
])


def main():
    for country in countries:
        csv_file = f'./data/translated_csv/{country}.csv'
        df = pd.read_csv(csv_file)
        cat1_set = set(df[category_name1].unique())
        cat2_set = set(df[category_name2].unique())
        cat3_set = set(df[category_name3].unique())
        print(f"{country} {category_name1} count: {len(cat1_set)}")
        print(f"{country} {category_name2} count: {len(cat2_set)}")
        print(f"{country} {category_name3} count: {len(cat3_set)}")

        # print(
        #     f"{country} {category_name} count: {len(df[category_name].unique())}")
        # category_names = list(df[category_name].value_counts().index)
        # current_category_set = set(category_names)
        # print(current_category_set - img_category_set)
        # if current_category_set.issubset(img_category_set):
        #     print(f"{country} is a subset of ph img_category_set")
        # else:
        #     print(f"{country} is not a subset of ph img_category_set")

        # category_count = list(df[category_name].value_counts())
        # for name, count in zip(category_names, category_count):
        #     print(f"{name},, {count}")

        print()
        print()


if __name__ == '__main__':
    main()
