import argparse
import gzip
import sys
from typing import Tuple

import numpy as np


DATA_DELIMITER = ','


def load_data(input_file: str, dtype=None) -> Tuple[str, np.ndarray]:
    if input_file.endswith('.gz'):
        with gzip.open(input_file, 'rt') as f:
            headers = f.readline().strip('# \n')
    else:
        with open(input_file, 'r') as f:
            headers = f.readline().strip('# \n')
    data = np.loadtxt(input_file, delimiter=DATA_DELIMITER)
    if dtype:
        data = np.asarray(data, dtype=dtype)
    print(f'Loaded {data.shape[0]} x {data.shape[1]} matrix with {data.size} '
          f'cells.')
    filled_cells = np.count_nonzero(data)
    print(f'Fill: {filled_cells} / {data.size} '
          f'({filled_cells / data.size * 100:.2f}%)')
    return headers, data


def fill_matrix(data: np.ndarray, symmetric: bool) -> None:
    updated_cells = 0
    for row in range(data.shape[0]):
        # Only iterate over the upper triangular.
        for col in range(row + 1, data.shape[1]):
            upper = data[row][col]
            lower = data[col][row]
            if upper == lower or (not symmetric and upper != 0 and lower != 0):
                continue
            # upper != lower
            updated_cells += 1
            if upper == 0:
                data[row][col] = lower
            elif lower == 0:
                data[col][row] = upper
            # upper != lower and upper != 0 and lower != 0
            # We reach this point only if symmetric == True, else we
            # would have continued above.
            elif lower < upper:
                data[row][col] = lower
            else:
                data[col][row] = upper
    print(f'Updated {updated_cells} '
          f'({updated_cells / data.size * 100:.2f}%) cells.')
    filled_cells = np.count_nonzero(data)
    print(f'New fill: {filled_cells} / {data.size} '
          f'({filled_cells / data.size * 100:.2f}%)')


def main() -> None:
    desc = """Fill the input matrix by mirroring values across the diagonal if
    entries are missing. For example, if there is in entry at position
    m[i][j] but not at m[j][i], the value from m[i][j] is copied. With the
    --symmetric parameter both m[i][j] and m[j][i] will be set to
    min(m[i][j], m[j][i]).
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('-s', '--symmetric',
                        action='store_true',
                        help='force symmetry by mirroring the smaller value '
                             'from the upper/lower triangle')
    parser.add_argument('-d', '--data-type',
                        help='specify data type of matrix (default: float)')
    args = parser.parse_args()
    symmetric = args.symmetric

    if not args.data_type or args.data_type == 'float':
        data_type = None
        data_fmt = '%.18e'  # numpy.savetxt default format
    elif args.data_type == 'int':
        data_type = int
        data_fmt = '%d'
    else:
        print(f'Invalid data type specified: {args.data_type}', file=sys.stderr)
        sys.exit(1)

    headers, data = load_data(args.input_file, dtype=data_type)
    fill_matrix(data, symmetric)

    if symmetric:
        if not np.allclose(data, data.T, equal_nan=True):
            print(f'Error: Output matrix is not symmetric!', file=sys.stderr)
            sys.exit(1)

    np.savetxt(args.output_file,
               data,
               delimiter=DATA_DELIMITER,
               header=headers,
               fmt=data_fmt)


if __name__ == '__main__':
    main()
    sys.exit(0)
