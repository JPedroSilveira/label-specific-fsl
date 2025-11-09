import functools
from typing import Any, List

class MatrixColumn:
    def __init__(self, values: List[Any]):
        self.values = values
    def __str__(self):
        return f"Column {self.index}:\n" + "\n".join(str(value) for value in self.values)
    

def sort_matrix_columns(matrix: List[List[Any]], reverse: bool = False):
    '''
    Order a matrix based on rows values,
    starting by the first row and using the others as tiebreakers.

    Conditions:
    - Matrix should be a list of lists where each list is a row;
    - Matrix should have all columns with the same size;
    - Matrix should have all rows with same size.
    '''
    n_rows = len(matrix)
    if n_rows == 0:
        return matrix
    n_columns = len(matrix[0])
    if n_columns == 0:
        return matrix
    columns: List[MatrixColumn] = []
    for column in range(0, n_columns):
        values = []
        for row in range(0, n_rows):
            values.append(matrix[row][column])
        columns.append(MatrixColumn(values=values))
    sorted_columns: List[MatrixColumn]  = sorted(columns, key=functools.cmp_to_key(_compare), reverse=reverse)
    ordered_matrix = []
    for row in range(0, n_rows):
        ordered_matrix.append([])
    for column in sorted_columns:
        for row_index, row_value in enumerate(column.values):
            ordered_matrix[row_index].append(row_value)
    return ordered_matrix
    
def _compare(a: MatrixColumn, b: MatrixColumn):
    for index, a_value in enumerate(a.values):
        b_value = b.values[index]
        if a_value > b_value: # When a has a value greater than b, it ends and a "wins"
            return 1
        if b_value > a_value: # When b has a value greater than a, it ends and b "wins"
            return -1
        # When a_value == b_value it should continue to the next value
    return 0 # If lists are identical