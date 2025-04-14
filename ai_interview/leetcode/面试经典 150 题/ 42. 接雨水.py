from typing import List

class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        if n <= 2:
            return 0
        
        left_max = [0] * n
        right_max = [0] * n
        left_max[0] = height[0]
        right_max[n-1] = height[n-1]

        # 从左往右遍历
        for i in range(1, n):
            left_max[i] = max(left_max[i-1], height[i])
        
        for j in range(n-2, -1, -1):
            right_max[j] = max(right_max[j+1], height[j])

        ans = 0
        for i in range(n):
            ans += min(left_max[i], right_max[i]) - height[i]
        
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