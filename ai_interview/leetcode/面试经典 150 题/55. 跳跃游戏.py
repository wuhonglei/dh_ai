from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if len(nums) <= 1:
            return True
        
        n = len(nums)
        max_dist = 0
        for i in range(0, n - 1):
            num = nums[i]
            if num == 0 and max_dist <= i:
                return False

            max_distance = i + num
            if max_distance >= n - 1:
                # 如果当前跳能直接到达，则直接返回 True
                return True
            else:
               max_dist = max(max_dist, max_distance)

        return False


case_list = [
    ([3,0,2,0,0,1,4], False),
    ([3,2,1,0,4], False),
    ([2,3,1,1,4], True),
]

for case_item in case_list:
    s = Solution()
    element = s.canJump(case_item[0])
    assert element == case_item[1], f'expect {case_item[1]}, but get {element}'
    print('success')