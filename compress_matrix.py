import argparse
import gzip
import itertools
import sys
from typing import Tuple

import numpy as np


DATA_DELIMITER = ','


def load_data(input_file: str, dtype=None) -> Tuple[list, np.ndarray]:
    if input_file.endswith('.gz'):
        with gzip.open(input_file, 'rt') as f:
            headers = f.readline().strip('# \n').split(DATA_DELIMITER)
    else:
        with open(input_file, 'r') as f:
            headers = f.readline().strip('# \n').split(DATA_DELIMITER)
    data = np.loadtxt(input_file, delimiter=DATA_DELIMITER)
    if dtype:
        data = np.asarray(data, dtype=dtype)
    print(f'Loaded {data.shape[0]} x {data.shape[1]} matrix with {data.size} '
          f'cells.')
    filled_cells = np.count_nonzero(data)
    print(f'Fill: {filled_cells} / {data.size} '
          f'({filled_cells / data.size * 100:.2f}%)')
    return headers, data


def compress_matrix(headers: list, data: np.ndarray) -> Tuple[list, np.ndarray]:
    if data.shape[0] != data.shape[1] or len(data.shape) != 2:
        print(f'Error: 2D square matrix required. Got: {data.shape}')
        return np.array()
    take_indizes = list()
    selectors = list()
    for idx in range(data.shape[0]):
        row_zero = np.allclose(data[idx], 0)
        col_zero = np.allclose(data[:, idx], 0)
        if row_zero or col_zero:
            selectors.append(0)
            continue
        selectors.append(1)
        take_indizes.append(idx)
    removed_pairs = data.shape[0] - len(take_indizes)
    print(f'Removed {removed_pairs} '
          f'({removed_pairs / data.shape[0] * 100:.2f}%) row-column pairs.')
    ix_grid = np.ix_(take_indizes, take_indizes)
    compressed_data = data[ix_grid]
    print(f'Compressed matrix to {compressed_data.size} '
          f'({compressed_data.size / data.size * 100:.2f}%) cells.')
    filled_cells = np.count_nonzero(compressed_data)
    print(f'New fill: {filled_cells} / {compressed_data.size} '
          f'({filled_cells / compressed_data.size * 100:.2f}%)')
    return list(itertools.compress(headers, selectors)), compressed_data



def main() -> None:
    desc = """Compress the input matrix by removing row-column pairs
    where either the row or the column contains only zero values.
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('-d', '--data-type',
                        help='specify data type of matrix (default: float)')
    args = parser.parse_args()

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
    headers, data = compress_matrix(headers, data)

    np.savetxt(args.output_file,
               data,
               delimiter=DATA_DELIMITER,
               header=DATA_DELIMITER.join(headers),
               fmt=data_fmt)


if __name__ == '__main__':
    main()
    sys.exit(0)
