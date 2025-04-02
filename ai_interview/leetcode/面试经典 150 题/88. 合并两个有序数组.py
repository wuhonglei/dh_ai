from typing import List

class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        p1, p2 = m - 1, n - 1
        tail = m + n - 1
        while p1 >= 0 or p2 >= 0:
            if p2 < 0:
                break

            if p1 < 0:
                nums1[tail] = nums2[p2]
                p2 -= 1
            elif nums1[p1] > nums2[p2]:
                nums1[tail] = nums1[p1]
                p1 -= 1
            else:
                nums1[tail] = nums2[p2]
                p2 -= 1
            tail -= 1

s = Solution()
num1 = [1,2,3,4,0, 0,0]
s.merge(num1, 4, [2,5,6], 3)
print(num1)