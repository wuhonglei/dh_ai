from typing import List, Dict

class Node:
    def __init__(self, value: int):
        self.value = value
        self.min_index = -1
        self.max_index = -1
        self.index_list: List[int] = []

class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans, nxt, big = [0] * n, {}, 10 ** 9
        for i in range(n-1, -1, -1):
            current_temp = temperatures[i]
            warmer_index = min(nxt.get(temp, big) for temp in range(current_temp +1, 101))
            if warmer_index != big:
                ans[i] = warmer_index - i
            nxt[current_temp] = i
        return ans


s = Solution()
print(s.dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73]))
# print(s.dailyTemperatures([30, 40, 50, 60]))
# print(s.dailyTemperatures([30, 60, 90]))
# print(s.dailyTemperatures([]))
