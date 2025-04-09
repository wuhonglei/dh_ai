from typing import List

class Solution:
    def hIndex(self, citations: List[int]) -> int:
        sorted_citation = sorted(citations) # 升序排列
        h = 0
        n = len(sorted_citation)
        for i, citation in enumerate(sorted_citation):
            count = n - i
            if count > h:
                h = min(citation, count)
        return h
    

case_list = [
    ([3,0,6,1,5], 3), # [0, 1, 3, 5, 6]
    ([1,3,1], 1)
]

for case_item in case_list:
    s = Solution()
    element = s.hIndex(case_item[0])
    assert element == case_item[1], f'expect {case_item[1]}, but get {element}'
    print('success')