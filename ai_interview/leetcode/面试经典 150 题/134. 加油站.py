from typing import List

class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)
        i = 0
        while i < n:
            sum_of_gas = sum_of_cost = 0
            cnt = 0
            while cnt < n:
                j = (i + cnt) % n
                sum_of_gas += gas[j]
                sum_of_cost += cost[j]
                if sum_of_cost > sum_of_gas:
                    break
                cnt += 1
            if cnt == n:
                return i
            else:
                i += cnt + 1
        return -1


case_list = [
    ([5,1,2,3,4], [4,4,1,5,1], 4),
    ([1,2,3,4,5], [3,4,5,1,2], 3),
    ([2,3,4], [3,4,3], -1)
]

for case_item in case_list:
    s = Solution()
    element = s.canCompleteCircuit(case_item[0], case_item[1])
    assert element == case_item[2], f'expect {case_item[1]}, but get {element}'
    print('success')