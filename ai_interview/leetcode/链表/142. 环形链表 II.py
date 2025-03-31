from typing import Optional, List
from utils import ListNode, create_list, print_list

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        visited = set()
        while head:
            if head in visited:
                return head
            
            visited.add(head)
            head = head.next

        return None
    

a = Solution()
b = create_list([1, 2, 3, 4], 1)
c = a.detectCycle(b)
print_list(c)