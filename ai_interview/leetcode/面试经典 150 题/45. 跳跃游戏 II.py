from typing import List

class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        max_position, end, steps = 0, 0, 0
        for i in range(n - 1):
            if max_position >= i:
                max_position = max(max_position, i + nums[i])
                if i == end:
                    end = max_position
                    steps += 1
        return steps



    def find_min_index(self, temp_nums: List[int]) -> int:
        last_index = len(temp_nums) - 1
        min_index = float('inf')
        i = last_index
        while i >=0:
            if i + temp_nums[i] >= last_index:
                min_index = i
            i -= 1

        return min_index


case_list = [
    ([2,3,1,1,4], 2),
    ([2,3,0,1,4], 2)
]

for case_item in case_list:
    s = Solution()
    element = s.jump(case_item[0])
    assert element == case_item[1], f'expect {case_item[1]}, but get {element}'
    print('success')