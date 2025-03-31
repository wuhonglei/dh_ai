from typing import List, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def create_list(nums: List[int], pos: int = -1) -> Optional[ListNode]:
    if not nums:
        return None

    node_list = []
    for num in nums:
        temp = ListNode(num)
        node_list.append(temp)


    head = None
    current = None
    for node in node_list:
        if not current:
            head = current = node
            continue
        current.next = node
        current = node

    if pos is None or pos == -1:
        return head
    
    node_list[-1].next = node_list[pos]
    return head


def print_list(head: Optional[ListNode]) -> None:
    curr = head
    walked = set()
    while curr:
        print(curr.val)
        walked.add(curr)
        curr = curr.next
        if curr in walked:
            break
