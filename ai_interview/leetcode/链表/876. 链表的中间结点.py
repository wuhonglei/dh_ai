from typing import Optional, List
from utils import ListNode, create_list, print_list

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while slow and fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        return slow


a = [1, 2, 3, 4, 5, 6]
b = create_list(a)
c = Solution()
print_list(c.middleNode(b))
