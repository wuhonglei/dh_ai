from typing import List
from collections import defaultdict

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        freq_dict = defaultdict(int)
        for num in nums:
            freq_dict[num] += 1
        limit = len(nums) / 2
        for num, freq in freq_dict.items():
            if freq > limit:
                return num
            
case_list = [
    ([3,2,3], 3),
    ([2,2,1,1,1,2,2], 2)
]

for case_item in case_list:
    s = Solution()
    element = s.majorityElement(case_item[0])
    assert element == case_item[1]
    print('success')