import pprint as pp
import pandas as pd
import requests

headers = {
    "user-agent": "Mozilla/5.0 (Macintosh Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36"
}

count = 0


class Crawler:
    def __init__(self) -> None:
        self.count = 0

    def get_category(self, keyword: str):
        url = f"https://shopee.com.mx/api/v4/search/search_filter_config?keyword={keyword}&page_type=search&scenario=PAGE_GLOBAL_SEARCH"
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            try:
                result = r.json()
                category = result["data"]["filter_configuration"][
                    "dynamic_filter_group_data"
                ]["facets"][0]["category"]["parent_category_detail"]["category"][
                    "display_name"
                ]
                print(keyword, category)
                return category
            except:
                self.count += 1
                print("------", keyword)
                return None
        else:
            print(f"Error: status code - {r.status_code}")
            return None

    def main(self, df: pd.DataFrame) -> pd.DataFrame:
        for keyword in df["Keyword_Filter"][:]:
            category = self.get_category(str(keyword))
            df.loc[df["Keyword_Filter"] == keyword, "Category_Crawler"] = category
        df.to_csv("crawler.csv", index=False)
        print(self.count)
        return df
