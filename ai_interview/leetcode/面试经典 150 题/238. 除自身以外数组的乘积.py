from typing import List

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        answer = [0] * n

        # L[i] = L[i-1] * nums[i-1]
        answer[0] = 1
        for i in range(1, n):
            answer[i] = answer[i-1] * nums[i-1]

        R = 1
        for i in range(n-1, -1, -1):
            answer[i] = answer[i] * R
            R = R * nums[i]

        return answer

case_list = [
    ([1,2,3,4], [24,12,8,6]),
    ([-1,1,0,-3,3], [0,0,9,0,0])
]

for case_item in case_list:
    s = Solution()
    element = s.productExceptSelf(case_item[0])
    assert element == case_item[1], f'expect {case_item[1]}, but get {element}'
    print('success')