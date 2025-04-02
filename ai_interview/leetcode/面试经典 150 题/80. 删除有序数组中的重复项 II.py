from typing import List

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return n
        slow = 1
        fast = 2
        while fast < len(nums):
            if nums[fast] != nums[slow-1]:
                slow += 1
                nums[slow] = nums[fast]
            fast += 1
        return slow + 1

s = Solution()
nums =  [1,1,1,2,2,3]
print( s.removeDuplicates(nums))
print(nums)