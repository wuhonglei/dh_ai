from typing import List


class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        # 1. 先排序
        sorted_points = sorted(points, key=lambda x: x[1])
        ans = 0
        pos = float('-inf')
        for i in range(len(sorted_points)):
            x, y = sorted_points[i]
            if x > pos:
                ans += 1
                pos = y

        return ans

s = Solution()
print(s.findMinArrowShots([[10,16],[2,8],[1,6],[7,12]]))
print(s.findMinArrowShots([[1,2],[3,4],[5,6],[7,8]]))
print(s.findMinArrowShots([[1,2],[2,3],[3,4],[4,5]]))