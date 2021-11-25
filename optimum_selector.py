import argparse
import gzip
import sys
from typing import Any, List, Tuple

import numpy as np
import numpy.ma as ma

from Selector import Selector

DATA_SUFFIX = '.csv.gz'
DATA_DELIMITER = ','

def read_input(input_file: str) -> Tuple[List[Any], np.ndarray]:
    with gzip.open(input_file, 'rt') as f:
        headers = f.readline().lstrip('#').strip().split(',')
    data = np.loadtxt(input_file, delimiter=DATA_DELIMITER)
    return headers, data


def min_mean_col(m: np.ndarray) -> int:
    """Calculate the index of the column with the smallest mean.
    """
    col_mean = np.mean(m, axis=0)
    return np.argmin(col_mean)


def min_inverse_rank_col_pref_large(m: ma.MaskedArray) -> int:
    """Calculate the index of the column with the smallest of column
    values weighted with their respective inverse rank, giving the
    largest value the highest rank.

    Notes
    -----
    First, order the column values in ascending order. Then assign
    weights in order 1/m, ..., 1/2, 1/1 (for an m x n matrix) to
    the column values. The largest value gets the largest weight.
    Calculate the sum of these weighted values and return the index
    of the column with the smallest sum.
    """
    # Sort by ascending column value, putting masked values at the
    # end.
    sdata = np.sort(m, axis=0)
    # Get the number of non-masked rows.
    rows = np.max(ma.count(m, axis=0))
    # Calculate weights in ascending order: 1/m, ..., 1/1
    weights = 1 / np.arange(rows, 0, -1)
    # Add zeros for masked rows to fit size.
    weights.resize(sdata.shape[1])
    # Apply weights columnwise.
    # [:,None] required to project 1-D array to column.
    sdata *= weights[:,None]
    min_col = np.argmin(np.sum(sdata, axis=0))
    return min_col


def min_inverse_rank_col_pref_small(m: ma.MaskedArray) -> int:
    """Calculate the index of the column with the smallest of column
    values weighted with their respective inverse rank, giving the
    smallest value the highest rank.

    Notes
    -----
    First, order the column values in ascending order. Then assign
    weights in order 1/1, 1/2, ..., 1/m (for an m x n matrix) to
    the column values. The smallest value gets the largest weight.
    Calculate the sum of these weighted values and return the index
    of the column with the smallest sum.
    """
    # Sort by ascending column value, putting masked values at the
    # end.
    sdata = np.sort(m, axis=0)
    # Get the number of non-masked rows.
    rows = np.max(ma.count(m, axis=0))
    # Calculate weights in descending order: 1/1, ..., 1/m
    weights = 1 / np.arange(1, rows + 1)
    # Add zeros for masked rows to fit size.
    weights.resize(sdata.shape[1])
    # Apply weights columnwise.
    # [:,None] required to project 1-D array to column.
    sdata *= weights[:,None]
    min_col = np.argmin(np.sum(sdata, axis=0))
    return min_col


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('-m', '--mask-value', type=float,
                        help='mask (ignore) value in input_file')
    args = parser.parse_args()

    input_file = args.input_file
    if not input_file.endswith(DATA_SUFFIX):
        print(f'Error: Expected {DATA_SUFFIX} file.', file=sys.stderr)
        sys.exit(1)

    output_file = args.output_file

    headers, data = read_input(input_file)
    selector = Selector(headers,
                        data,
                        score=min_inverse_rank_col_pref_small,
                        summary=np.mean,
                        mask_value=args.mask_value)
    selector.process()
    with gzip.open(output_file, 'wt') as o:
        for line in selector.steps:
            o.write(DATA_DELIMITER.join(map(str, line)) + '\n')


if __name__ == '__main__':
    main()
    sys.exit(0)
