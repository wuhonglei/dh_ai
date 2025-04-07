from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 单调递减的栈
        min_buy_price = float('inf')
        max_profit = 0
        for price in prices:
            profit = price - min_buy_price
            if profit < 0:
                # 当前股票价格比较低, 买入
                min_buy_price = price
            elif profit > max_profit:
                max_profit = profit

        return max_profit


case_list = [
    ([7,1,5,3,6,4], 5),
    ([7,6,4,3,1], 0),
    ([4,2,3,1,0,8], 8)
]

for case_item in case_list:
    s = Solution()
    element = s.maxProfit(case_item[0])
    assert element == case_item[1]
    print('success')