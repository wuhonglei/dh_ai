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
    scale = -2
    special_char_to_int = {
        "IV": scale * char_to_int["I"],
        "IX": scale * char_to_int["I"],
        "XL": scale * char_to_int["X"],
        "XC": scale * char_to_int["X"],
        "CD": scale * char_to_int["C"],
        "CM": scale * char_to_int["C"],
    }
    def romanToInt(self, s: str) -> int:
        ans = 0
        last_char = ""
        for c in s:
            ans += Solution.char_to_int[c]
            concat = last_char + c
            if concat in Solution.special_char_to_int:
                ans += Solution.special_char_to_int[concat]
                last_char = ""
            else:
                last_char = c
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