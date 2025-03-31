from typing import Optional, List
from utils import ListNode, create_list, print_list


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pre = None
        curr = head
        while curr:
            temp = curr.next
            curr.next = pre
            pre = curr
            curr = temp

        return pre


a = [1, 2, 3, 4, 5]
b = create_list(a)
c = Solution()
d = c.reverseList(b)
print_list(d)
