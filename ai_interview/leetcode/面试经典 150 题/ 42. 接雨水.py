from typing import List

class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        if n <= 2:
            return 0
        
        left_max = [0] * n
        left_max[0] = height[0]

        # 从左往右遍历
        for i in range(1, n):
            left_max[i] = max(left_max[i-1], height[i])
        
        right_max = height[n-1]
        ans = 0
        for j in range(n-2, -1, -1):
            right_max = max(right_max, height[j])
            ans += min(right_max, left_max[j]) - height[j]
        
        return ans

case_list = [
    ([0,1,0,2,1,0,1,3,2,1,2,1], 6),
    ([4,2,0,3,2,5], 9)
]

for case_item in case_list:
    s = Solution()
    element = s.trap(case_item[0])
    assert element == case_item[1], f'expect {case_item[1]}, but get {element}'
    print('success')