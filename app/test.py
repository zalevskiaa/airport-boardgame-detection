import unittest

from solution_finder.solution_finder import SolutionFinder


class TestSolutionFinder(unittest.TestCase):
    test_cases = [
        (
            {
                'level': 'starter',
                'matrix': [
                    ['d', '.', '.', 'r'],
                    ['.', '.', 'u', '.'],
                    ['l', '.', '.', '.'],
                    ['.', 'r', '.', 'u'],
                ],
                'roads': None,
            },
            [
                ['owo', 'gwg', 'bwb', 'bwb'],
                ['owo', 'gwg', 'gwg', 'bwb'],
                ['rwr', 'wrw', 'wbw', 'wbw'],
                ['rwr', 'wrw', 'wrw', 'wbw'],
            ]
        ),
        (
            {
                'level': 'junior',
                'matrix': [
                    ['u', '.', '.', 'l'],
                    ['u', '.', '.', '.'],
                    ['.', '.', 'l', '.'],
                    ['.', 'l', '.', 'u'],
                ],
                'roads': [
                    [(0, 0), (1, 0), (2, 2), (3, 3)],
                    [(0, 3)],
                    [(3, 1)],
                ]
            },
            [
                ['gwg', 'gwg', 'wrw', 'wrw'],
                ['wbw', 'gwg', 'rwr', 'wrw'],
                ['wbw', 'wbw', 'rwr', 'bwb'],
                ['owo', 'owo', 'bwb', 'bwb'],
            ]
        ),
        (
            {
                'level': 'junior',
                'matrix': [
                    ['u', '.', '.', 'l'],
                    ['u', '.', '.', 'r'],
                    ['.', '.', '.', '.'],
                    ['l', '.', '.', 'd']
                ],
                'roads': [
                    [(0, 0), (1, 0)],
                    [(0, 3), (1, 3)],
                    [(3, 3)],
                    [(3, 0)]
                ]
            },
            [
                ['rwr', 'rwr', 'owo', 'owo'],
                ['wrw', 'wrw', 'bwb', 'bwb'],
                ['wrw', 'wbw', 'gwg', 'bwb'],
                ['wbw', 'wbw', 'gwg', 'gwg'],
            ]
        ),
    ]

    @staticmethod
    def str_mat(mat):
        if mat is None:
            return 'None'
        result = []
        for row in mat:
            result.append(' '.join(map(str, row)))
        return '\n'.join(result)

    def test_solution_finder(self):
        sf = SolutionFinder()

        for input_data, expected in self.test_cases:
            with self.subTest(input_data=input_data):
                result = sf.find_solution(input_data)
                self.assertEqual(
                    result, expected, (
                     f"Failed for {input_data}.\n" +
                     f"Expected:\n{self.str_mat(expected)}\n" +
                     f"Got:\n{self.str_mat(result)}")
                )


if __name__ == '__main__':
    unittest.main()
