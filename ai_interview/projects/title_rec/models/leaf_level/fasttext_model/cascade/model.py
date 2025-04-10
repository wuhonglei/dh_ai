"""
使用 fasttext 先预测一级目录，然后预测一级目录下的叶子目录
"""

import os
import time
import fasttext
import pandas as pd
from typing import Any
