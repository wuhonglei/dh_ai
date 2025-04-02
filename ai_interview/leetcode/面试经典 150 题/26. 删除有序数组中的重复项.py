from typing import List

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0

        slow = fast = 1
        n = len(nums)
        while fast < n:
            if nums[fast] != nums[fast - 1]:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1

        return slow

s = Solution()
nums =  [0,0,1,1,1,2,2,3,3,4]
print(s.removeDuplicates(nums))
print(nums)