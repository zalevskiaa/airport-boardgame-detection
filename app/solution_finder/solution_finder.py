class Direction:
    directions: list = ['u', 'r', 'd', 'l']

    def __init__(self, d: int | str):
        if isinstance(d, int):
            if not 0 <= d < len(Direction.directions):
                raise ValueError

            self.direction_index = d

        elif isinstance(d, str):
            if d not in Direction.directions:
                raise ValueError

            self.direction_index = Direction.directions.index(d)

        else:
            raise TypeError

    def __repr__(self):
        return f'Direction({Direction.directions[self.direction_index]})'

    def __eq__(self, value: 'Direction'):
        return self.direction_index == value.direction_index

    def copy(self):
        return Direction(self.direction_index)

    def rotate(self, times: int):
        self.direction_index = \
            (self.direction_index + times) % len(Direction.directions)

    def rotated(self, times):
        result = self.copy()
        result.rotate(times)
        return result

    def rotated_right(self):
        return self.rotated(1)

    def rotated_opposite(self):
        return self.rotated(2)

    def rotated_left(self):
        return self.rotated(3)


class Piece:
    def __init__(self,
                 color: str,
                 direction: str | Direction,
                 pad_cells: list[tuple[int, int]]
                 ):
        '''
        color - unique str
        direction - one of ('u', 'r', 'd', 'l')
        pad_cells - [(y1, x1), ...], where y1, x1 - coords relative to plane
        '''
        self.color: str = color
        if isinstance(direction, str):
            self.plane_direction: Direction = Direction(direction)
        elif isinstance(direction, Direction):
            self.plane_direction: Direction = direction.copy()
        self.pad_cells: list[tuple[int, int]] = pad_cells

    def __repr__(self):
        return f'<Piece "{self.color}">'

    def copy(self):
        return Piece(self.color,
                     self.plane_direction.copy(),
                     self.pad_cells.copy())

    def rotate(self, times=1):
        self.plane_direction.rotate(times)
        for _ in range(times):
            for i, (y, x) in enumerate(self.pad_cells):
                self.pad_cells[i] = (x, -y)

    def rotated(self, times):
        result = self.copy()
        result.rotate(times)
        return result

    def rotated_right(self):
        return self.rotated(1)

    def rotated_opposite(self):
        return self.rotated(2)

    def rotated_left(self):
        return self.rotated(3)

    def rotated_to(self, direction: Direction):
        times = (
            direction.direction_index
            + len(Direction.directions)
            - self.plane_direction.direction_index
        ) % len(Direction.directions)

        return self.rotated(times)


class NodeCell:
    def __init__(self,
                 possible_directions: list[Direction] = [],
                 covered: bool = False,
                 covered_color: str | None = None,
                 road_index: int = None,
                 ):
        self.possible_directions: list[Direction] = possible_directions.copy()
        self.covered: bool = covered
        self.covered_color: str | None = covered_color
        self.road_index: int = road_index

    def copy(self) -> 'NodeCell':
        return NodeCell([d.copy() for d in self.possible_directions],
                        self.covered, self.covered_color, self.road_index)


class Node:
    size = 4

    def __init__(self, matrix=None, available_pieces=None, roads=None):
        '''
        A node is represented as a matrix 4x4. for each cell we have:
        - list of possible planes direcitons
        - bool value whether cell is covered by a piece

        We also must remember which pieces were not used yet
        We also store same-road cells' coordinates
        '''
        self.matrix: list[list[NodeCell]] = \
            [[NodeCell()] * Node.size for _ in range(Node.size)]
        self.available_pieces: list[Piece] = []
        self.roads: list[tuple[int, int]] = []

        if matrix is not None:
            self.matrix = matrix
        if available_pieces is not None:
            self.available_pieces = available_pieces
        if roads is not None:
            self.roads = roads

    def __repr__(self):
        result = []
        for row in self.matrix:
            line = []
            for elem in row:
                line.append(elem.covered_color
                            if elem.covered_color else '...')
            line = ' '.join(line)
            result.append(line)
        result = '\n'.join(result)
        return f'<Node\n{result}>'

    def copy(self) -> 'Node':
        result = Node()

        for i, row in enumerate(self.matrix):
            for j, elem in enumerate(row):
                result.matrix[i][j] = elem.copy()

        result.available_pieces = [p.copy() for p in self.available_pieces]
        result.roads = [r for r in self.roads]

        return result

    def can_put_piece(self, piece: Piece, i: int, j: int):
        for pi, pj in piece.pad_cells + [(0, 0)]:
            if not 0 <= i + pi < Node.size or not 0 <= j + pj < Node.size:
                return False
            if self.matrix[i + pi][j + pj].covered:
                return False
        return True

    def with_put_piece(self, piece: Piece, i: int, j: int):
        result = self.copy()

        result.matrix[i][j].covered = True
        result.matrix[i][j].covered_color = piece.color

        for pi, pj in piece.pad_cells:
            result.matrix[i + pi][j + pj].covered = True
            result.matrix[i + pi][j + pj].covered_color = piece.color

        # todo: modify possible directions along roads

        return result


def make_node_from_input_data(matrix, roads, pieces):
    node = Node()

    for i, row in enumerate(matrix):
        for j, elem in enumerate(row):
            if elem == '.':
                node.matrix[i][j] = NodeCell([])
            else:
                d = Direction(elem)

                # rodated for junior and master levels
                possible_directions = \
                    [d, d.rotated_opposite()] if roads is not None else [d]
                node.matrix[i][j] = NodeCell(
                    possible_directions=possible_directions
                )

    node.available_pieces = [p.copy() for p in pieces]

    if roads is not None:
        # for junior and master levels
        for road_index, cells_indices in enumerate(roads):
            for i, j in cells_indices:
                node.matrix[i][j].road_index = road_index
        node.roads = roads

    return node


class SolutionFinder:
    pieces = [
        Piece('owo', 'd', [(1, 0)]),
        Piece('gwg', 'u', [(0, -1), (-1, -1)]),
        Piece('bwb', 'r', [(1, 0), (0, -1)]),
        Piece('rwr', 'l', [(1, 0)]),
        Piece('wrw', 'r', [(-1, 0), (0, 1)]),
        Piece('wbw', 'u', [(-1, 0), (-1, -1)]),
    ]

    def _rec_find_solution(self, node: Node) -> Node | None:
        if not node.available_pieces:
            return node

        piece = node.available_pieces.pop(0)
        for i, row in enumerate(node.matrix):
            for j, elem in enumerate(row):
                for pd_index, possible_direction \
                        in enumerate(elem.possible_directions):

                    rotated_piece = piece.rotated_to(possible_direction)

                    if node.can_put_piece(rotated_piece, i, j):
                        new_node = node.with_put_piece(rotated_piece, i, j)

                        if len(elem.possible_directions) > 1:
                            road_index = new_node.matrix[i][j].road_index
                            road_cells_indices = new_node.roads[road_index]
                            for rci, rcj in road_cells_indices:
                                new_node.matrix[rci][rcj] \
                                    .possible_directions = \
                                    [new_node.matrix[rci][rcj]
                                     .possible_directions[pd_index]]

                        result = self._rec_find_solution(new_node)
                        if result is not None:
                            return result

        return None

    def find_solution(self, field_data: dict) -> Node | None:
        node = make_node_from_input_data(
            field_data['matrix'],
            field_data['roads'],
            SolutionFinder.pieces
        )

        assert field_data['level'] in ['junior', 'starter']

        result = self._rec_find_solution(node)

        if result is None:
            return result

        result_matrix = []
        for row in result.matrix:
            result_row = []
            for elem in row:
                result_row.append(elem.covered_color)
            result_matrix.append(result_row)
        return result_matrix


def str_mat(mat):
    result = []
    for row in mat:
        result.append(' '.join(map(str, row)))
    return '\n'.join(result)


def test():
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

    test_case = test_cases[2]

    field_data, expected = test_case

    sf = SolutionFinder()
    result = sf.find_solution(field_data)
    print(str_mat(result))

    assert result == expected


if __name__ == '__main__':
    test()
