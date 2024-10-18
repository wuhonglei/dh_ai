# Keyword-Categories
## Predict Accuracy

The ML model predicts the keyword category, and the local team then reviews the category. The following are the accuracies of the predictions.

| Country | 2023/2 | 2023/1 | 2022/12 | 2022/11 | 2022/10 |
| ------- | ------ | ------ | ------- | ------- | ------- |
| SG      | 88%    | 86%    | 88%     | 88%     | 87%     |
| MY      | 86%    | 87%    | 81%     | 87%     | 80%     |
| TH      | 71%    | 71%    | 73%     | 75%     | 85%     |
| TW      | 88%    | 86%    | 86%     | 83%     | 76%     |
| ID      | 61%    | 92%    | 72%     | 92%     | 86%     |
| VN      | 79%    | 91%    | 72%     | 71%     | 60%     |
| PH      | 89%    | 93%    | 88%     | 93%     | 85%     |
| BR      | 82%    | 71%    | 85%     | 88%     | 93%     |
| MX      | 74%    | 81%    | 78%     | 81%     | 80%     |
| CL      | 72%    | 65%    | 73%     | 71%     | 74%     |
| Average | 79%    | 82%    | 80%     | 83%     | 81%     |

## Getting Started
* install with pip
    ```
    $ python3 -v venv venv
    $ source venv/bin/activate
    $ pip3 install -r requirement.txt
    ```
* install with poetry
    ```
    $ poetry install
    $ poetry shell
    ```

## Usage

* run the code
    ```
    $ python3 main.py

    ```

* log
    ```
    [nltk_data] Downloading package stopwords to /Users/max/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    ------SG------
    03/21/2023 05:19:00 PM INFO <function get_gsheet_df> - Time Coast: 1.91s
    03/21/2023 05:19:01 PM INFO <function pre_process> - Time Coast: 1.03s
    03/21/2023 05:19:03 PM INFO <function train_model> - Time Coast: 2.59s
    03/21/2023 05:19:04 PM INFO <function update_gsheet_df> - Time Coast: 0.51s
    03/21/2023 05:19:04 PM INFO <function KeywordCategories.main> - Time Coast: 6.08s
    ------MY------
    03/21/2023 05:19:06 PM INFO <function get_gsheet_df> - Time Coast: 2.13s
    ...

    ```
