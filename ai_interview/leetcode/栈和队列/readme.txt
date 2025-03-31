给定 pushed 和 popped 两个序列，每个序列中的 值都不重复，只有当它们可能是在最初空栈上进行的推入 push 和弹出 pop 操作序列的结果时，返回 true；否则，返回 false 。

 

示例 1：

输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
输出：true
解释：我们可以按以下顺序执行：
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
示例 2：

输入：pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
输出：false
解释：1 不能在 2 之前弹出。

1. 初始化待入栈的 queue 为 pushed 序列
2. (从左往右)遍历 queue
  a. 将当前元素入 pushed 序列
  b. 如果 pushed 序列 和 popped 栈顶元素相同，则同时弹出 2 个序列的栈顶元素
3. queue 为空时，结束遍历
  a. 若 pushed 和 popped 都为空，表示顺序合法
  b. 若 pushed 和 popped 不都为空，表示顺序非法