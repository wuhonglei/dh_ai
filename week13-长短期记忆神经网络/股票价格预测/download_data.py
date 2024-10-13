"""
下载股票数据并保持到 stock_data.pkl 文件中
"""

from datetime import datetime
from pandas_datareader import data as pdr
import yfinance as yf
from pandas_datareader.data import DataReader
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
# %matplotlib inline

# For reading stock data from yahoo

# yf.pdr_override()

# For time stamps


# Set up End and Start times for data grab
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

# end = datetime.now()
# start = datetime(end.year - 1, end.month, end.day)

start='2012-01-01'
end='2023-01-31'

for stock in tech_list:
    globals()[stock] = yf.download(stock, start, end)


company_list = [AAPL, GOOG, MSFT, AMZN]
company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name

df = pd.concat(company_list, axis=0)
with open("stock_data.pkl", "wb") as f:
    pickle.dump(df, f)