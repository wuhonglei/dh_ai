class Solution:
    char_to_int = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000
    }
    def romanToInt(self, s: str) -> int:
        ans = 0
        last_char = ""
        n = len(s)
        for i, c in enumerate(s):
            value = Solution.char_to_int[c]
            if i < n - 1 and value <  Solution.char_to_int[s[i + 1]]:
                ans -= value
            else:
                ans += value
        return ans

case_list = [
    ("III", 3),
    ("IV", 4),
    ("IX", 9),
    ("LVIII", 58),
    ("MCMXCIV", 1994)
]

for case_item in case_list:
    s = Solution()
    element = s.romanToInt(case_item[0])
    assert element == case_item[1], f'expect {case_item[1]}, but get {element}'
    print('success')