from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        n = len(prices)
        for index in range(1, n):
            temp_profit = prices[index] - prices[index - 1]
            if temp_profit > 0:
                profit += temp_profit

        return profit


case_list = [
    ([7,1,5,3,6,4], 7),
    ([1,2,3,4,5], 4),
    ([7,6,4,3,1], 0)
]

for case_item in case_list:
    s = Solution()
    element = s.maxProfit(case_item[0])
    assert element == case_item[1], f'expect {case_item[1]}, but get {element}'
    print('success')