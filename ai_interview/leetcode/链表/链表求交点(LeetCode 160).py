from typing import Optional, List, Tuple


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def create_list(nums: List[int]) -> Optional[ListNode]:
    if not nums:
        return None

    head = None
    current = None
    for num in nums:
        temp = ListNode(num)
        if not current:
            head = current = temp
            continue
        current.next = temp
        current = temp

    return head


class Solution1:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        new_head_a, new_head_b = self.swap_list_node(headA, headB)
        return self.find_intersection_node(new_head_a, new_head_b)

    def get_list_len(self, head: ListNode) -> int:
        if not head:
            return 0
        _len = 0
        curr = head
        while curr:
            _len += 1
            curr = curr.next

        return _len

    def swap_list_node(self, headA: ListNode, headB: ListNode) -> Tuple[ListNode, ListNode]:
        len_a = self.get_list_len(headA)
        len_b = self.get_list_len(headB)
        short_head, long_head = (
            headA, headB) if len_a < len_b else (headB, headA)
        new_long_head = self.move_forward(
            long_head, max(len_a, len_b) - min(len_a, len_b))
        return short_head, new_long_head

    def move_forward(self, head: ListNode, step: int) -> ListNode:
        curr = head
        while step and curr:
            step -= 1
            curr = curr.next
        return curr  # type: ignore

    def find_intersection_node(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        curr_a = headA
        curr_b = headB
        while curr_a and curr_b:
            if curr_a == curr_b:
                return curr_a
            curr_a = curr_a.next
            curr_b = curr_b.next

        return None


class Solution2:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        set_a = set()
        temp = headA
        while temp:
            set_a.add(temp)
            temp = temp.next

        temp = headB
        while temp:
            if (temp in set_a):
                return temp
            temp = temp.next

        return None


class Solution3:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        if not headA or not headB:
            return None

        pa = headA
        pb = headB
        while pa != pb:
            pa = headB if not pa else pa.next
            pb = headA if not pb else pb.next

        return pa
