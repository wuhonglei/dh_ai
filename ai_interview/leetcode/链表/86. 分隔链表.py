from typing import Optional, List
from utils import ListNode, create_list, print_list


class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        less_head = ListNode(-1)
        pre_less = less_head

        larger_head = ListNode(-1)
        pre_larger = larger_head

        curr = head
        while curr:
            if curr.val < x:
                pre_less.next = curr
                pre_less = curr
            else:
                pre_larger.next = curr
                pre_larger = curr
            curr = curr.next

        pre_larger.next = None
        pre_less.next = larger_head.next
        return less_head.next


head = create_list([1, 4, 3, 2, 5, 2])
x = 3


b = Solution()
new_head = b.partition(head, x)
print_list(new_head)
