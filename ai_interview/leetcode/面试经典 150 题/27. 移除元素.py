from typing import List

class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            if nums[left] != val:
                left += 1
            else:
                nums[left] = nums[right]
                right -= 1
        return left


s = Solution()
num1 = [0,1,2,2,3,0,4,2]
k = s.removeElement(num1, 2)
print(k, num1)