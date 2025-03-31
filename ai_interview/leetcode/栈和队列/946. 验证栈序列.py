from typing import List


class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        st, j = [], 0
        for val in pushed:
            st.append(val)
            while st and st[-1] == popped[j]:
                st.pop()
                j += 1

        return len(st) == 0


s = Solution()
print(s.validateStackSequences([1, 2, 3, 4, 5], [4, 5, 3, 2, 1]))
print(s.validateStackSequences([1, 2, 3, 4, 5], [4, 3, 5, 1, 2]))
