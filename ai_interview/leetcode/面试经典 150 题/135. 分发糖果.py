from typing import List

class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        # 左规则, 右侧大于左侧时 +1, 否则置为 1
        left = [0] * n
        for i in range(n):
            if i > 0 and ratings[i] > ratings[i-1]:
                left[i] = left[i-1] + 1
            else:
                left[i] = 1

        right = total = 0
        # 右规则, 左侧大于右侧时 +1, 否则置为 1
        for i in range(n-1, -1, -1):
            if i < n - 1 and ratings[i] >ratings[i+1]:
                right = right + 1
            else:
                right = 1

            total += max(left[i], right)
        
        return total


case_list = [
    ([1,3,2,2,1], 7),
    ([1,0,2], 5),
    ([1,2,2], 4)
]

for case_item in case_list:
    s = Solution()
    element = s.candy(case_item[0])
    assert element == case_item[1], f'expect {case_item[1]}, but get {element}'
    print('success')