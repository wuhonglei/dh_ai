from typing import Optional, List
from utils import ListNode, create_list, print_list

# 队尾插入，队头删除
"""
1. 初始 head 和 rear 指向一个空的头指针 -1 位置。length = 3; current = 0;
2. 插入一个元素时，先判断队列是否满，没满时 (rear_post + 1) % length, current++
3. 删除一个元素，判断队列是否空，没空时，删除 head_pos 位置的元素，current--; 如果 current = 0，表示队列全空，此时重置队列
"""


class MyCircularQueue:

    def __init__(self, k: int):
        self.total = k+1
        self.head = self.rear = 0
        self.queue = [0] * (k + 1)

    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        self.queue[self.rear] = value
        self.rear = (self.rear + 1) % self.total
        return True

    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        self.head = (self.head + 1) % self.total
        return True

    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[self.head]

    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[(self.rear - 1 + self.total) % self.total]

    def isEmpty(self) -> bool:
        return self.head == self.rear

    def isFull(self) -> bool:
        return (self.rear + 1) % self.total == self.head

    # Your MyCircularQueue object will be instantiated and called as such:
    # obj = MyCircularQueue(k)
    # param_1 = obj.enQueue(value)
    # param_2 = obj.deQueue()
    # param_3 = obj.Front()
    # param_4 = obj.Rear()
    # param_5 = obj.isEmpty()
    # param_6 = obj.isFull()


circularQueue = MyCircularQueue(3)
print(circularQueue.enQueue(1))  # 返回 true
print(circularQueue.enQueue(2))  # 返回 true
print(circularQueue.enQueue(3))  # 返回 true
print(circularQueue.enQueue(4))  # 返回 false，队列已满
print(circularQueue.Rear())  # 返回 3
print(circularQueue.isFull())  # 返回 true
print(circularQueue.deQueue())  # 返回 true
print(circularQueue.enQueue(4))  # 返回 true
print(circularQueue.Rear())  # 返回 4
