from src.utils import get_gsheet_df, timer

import pandas as pd
import json
import openai
import time


class ChatGPTKeywordCategories:
    def __init__(self) -> None:
        self.api_key = [
            "your api key",
        ]

    def get_openapi_result(self, keyword_list):

        for i in range(0, len(keyword_list), 10):
            num = (i // 10) % 3
            openai.api_key = self.api_key[num]
            keywods = keyword_list[i : i + 10]
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"here is a keyword {str(keywods)}could you help to reply keyword is which category in below, format should be "
                + '[{"keyword":"category"},{"keyword":"category"},{"keyword":"category"}]'
                + "Beauty,Computing,Electronics,Fashion Accessories,Health,Home & Living,Home Appliance,Jewelry & Watches,Men Bags,Men Clothes,Men Shoes,Mobile & Gadgets,Mom & Baby,Others,Pets,Platform Terms (Mercado/Shopee),Sports & Fitness,Stationery,Toys & Hobbies,Travel & Luggage,Vehicle Accessories,Video Games,Women Bags,Women Clothes,Women Shoes,Food and Beverage",
                temperature=0.7,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

            result = response["choices"][0]["text"].replace("\n", "")
            print(i, response, result)
            data = json.loads(result)
            print("\n\n\n")
            result = []

            for item in data:
                name = list(item.keys())[0]
                category = item[name]
                result.append({"keyword": name, "category": category})
            df = pd.DataFrame(result)
            df.to_csv("./csv/chatgpt_result.csv", index=False, mode="a", header=False)

    @timer
    def main(self):
        # Step1 > Get gsheet raw data
        df = get_gsheet_df("CO")
        data = df[df["Category"] == ""]["Keyword"].tolist()

        # Step2 > Get openapi result
        self.get_openapi_result(data[:])


if __name__ == "__main__":

    categories = ChatGPTKeywordCategories()
    categories.main()
