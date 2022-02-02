import argparse
import gzip
import sys
from collections import Counter
from typing import Any, Callable, List, Tuple

import numpy as np
import numpy.ma as ma

from Selector import Selector


DATA_SUFFIX = '.csv.gz'
DATA_DELIMITER = ','

counters = list()
squared_matrix = None

def read_input(input_file: str, dtype = None) -> Tuple[List[Any], np.ndarray]:
    with gzip.open(input_file, 'rt') as f:
        headers = f.readline().lstrip('#').strip().split(',')
    data = np.loadtxt(input_file, delimiter=DATA_DELIMITER)
    if dtype:
        return headers, np.asarray(data, dtype=dtype)
    return headers, data


def min_mean_col(m: ma.MaskedArray) -> int:
    """Calculate the index of the column with the smallest mean.
    """
    if ma.count_masked(m) == m.size:
        return -1
    col_mean = np.nanmean(m, axis=0)
    return np.argmin(col_mean)


def max_mean_col(m: ma.MaskedArray) -> int:
    """Calculate the index of the column with the largest mean.
    """
    if ma.count_masked(m) == m.size:
        return -1
    col_mean = np.nanmean(m, axis=0)
    return np.argmax(col_mean)


def inverse_rank_col_pref_large(m: ma.MaskedArray) -> np.ndarray:
    """Calculate the sum of column values weighted with their
    respective inverse rank, giving the largest value the highest
    rank.

    Notes
    -----
    First, order the column values in ascending order. Then assign
    weights in order 1/m, ..., 1/2, 1/1 (for an m x n matrix) to
    the column values. The largest value gets the largest weight.
    Return the sum of these weighted values.
    """
    # Sort by ascending column value, putting masked values at the
    # end.
    sdata = np.sort(m, axis=0)
    # Get the number of non-masked rows per column.
    rows = ma.count(m, axis=0)
    # No unmasked values left.
    if all([row_count == 0 for row_count in rows]):
        return -1
    # If some value is masked, each column can contain a different number of
    # masked values. Since we want to apply the factor 1 to the last
    # non-masked value, we need to know its position. Therefore, we need to
    # compute the weights per column.
    for col, row_count in enumerate(rows):
        # Calculate weights in ascending order: 1/m, ..., 1/1
        weights = 1 / np.arange(row_count, 0, -1)
        # Add zeros for masked rows to fit size.
        weights.resize(sdata.shape[1])
        # Apply weights columnwise.
        sdata[:, col] *= weights
    return np.sum(sdata, axis=0)


def min_inverse_rank_col_pref_large(m: ma.MaskedArray) -> int:
    """Calculate the index of the column with the smallest sum of
    values weighted with their respective inverse rank, giving the
    largest value the highest rank.
    """
    return np.argmin(inverse_rank_col_pref_large(m))


def max_inverse_rank_col_pref_large(m: ma.MaskedArray) -> int:
    """Calculate the index of the column with the largest sum of
    values weighted with their respective inverse rank, giving the
    largest value the highest rank.
    """
    return np.argmax(inverse_rank_col_pref_large(m))


def inverse_rank_col_pref_small(m: ma.MaskedArray) -> np.ndarray:
    """Calculate the sum of column values weighted with their
    respective inverse rank, giving the smallest value the highest
    rank.

    Notes
    -----
    First, order the column values in ascending order. Then assign
    weights in order 1/1, 1/2, ..., 1/m (for an m x n matrix) to
    the column values. The smallest value gets the largest weight.
    Return the sum of these weighted values.
    """
    # Sort by ascending column value, putting masked values at the
    # end.
    sdata = np.sort(m, axis=0)
    # Get the number of non-masked rows.
    rows = np.max(ma.count(m, axis=0))
    # No unmasked values left.
    if rows == 0:
        return -1
    # Calculate weights in descending order: 1/1, ..., 1/m
    weights = 1 / np.arange(1, rows + 1)
    # Add zeros for masked rows to fit size.
    weights.resize(sdata.shape[1])
    # Apply weights columnwise.
    # [:,None] required to project 1-D array to column.
    sdata *= weights[:,None]
    return np.sum(sdata, axis=0)


def min_inverse_rank_col_pref_small(m: ma.MaskedArray) -> int:
    """Calculate the index of the column with the smallest sum of
    values weighted with their respective inverse rank, giving the
    smallest value the highest rank.
    """
    return np.argmin(inverse_rank_col_pref_small(m))


def max_inverse_rank_col_pref_small(m: ma.MaskedArray) -> int:
    """Calculate the index of the column with the largest sum of
    values weighted with their respective inverse rank, giving the
    smallest value the highest rank.
    """
    return np.argmax(inverse_rank_col_pref_small(m))


def max_weighted_dist(m: np.ndarray) -> int:
    """Calculate the index of the column with the largest
    weighted-distribution score.

    Notes
    -----
    The weighted distribution makes most sense for discrete
    distributions.
    For each column, let `n` be the total number of values and `o`
    the number of occurrences of value `x`. The score of value `x`
    is calculated as:
        `1 / x * o / n`
    The scores for all unique values in the column are calculated and
    summed up to form the overall score of the column. The column
    with the largest score is selected for removal.

    The final score is in the interval (0,1] and close to 1 if the
    column contains many small values.
    """
    global counters
    if not counters:
        print('Initializing counters')
        for col in range(m.shape[1]):
            total_value_count = ma.count(m[:, col])
            if total_value_count == 0:
                counters.append(Counter())
                continue
            values, counts = np.unique(m[:, col], return_counts=True)
            counters.append(Counter({value: count
                                     for value, count in zip(values, counts)
                                     if value is not ma.masked}))

    column_scores = list()
    valid_column_found = False
    for counter in counters:
        total_value_count = sum(counter.values())
        if total_value_count == 0:
            column_scores.append(np.nan)
            continue
        valid_column_found = True
        column_scores.append(np.sum([1 / value * count / total_value_count
                             for value, count in counter.items()]))
    if not valid_column_found:
        return -1
    rem_idx = np.nanargmax(column_scores)

    # Decrement counters of other columns.
    for counter_idx, removed_value in enumerate(m[rem_idx, :]):
        if removed_value is ma.masked:
            continue
        counters[counter_idx][removed_value] -= 1
    # Clear counter for removed column.
    counters[rem_idx] = Counter()
    return rem_idx


def max_square_sum(m: np.ndarray) -> int:
    """Calcuate the index of the column with the largest sum of
    squared values.

    Notes
    -----
    Should be used for values in interval [0,1]. Squaring these
    values implicitly gives a larger weight to values closer to 1.
    """
    global squared_matrix
    if squared_matrix is None:
        print('Squaring matrix.')
        squared_matrix = ma.masked_array(m ** 2)
        squared_matrix.mask = ma.make_mask_none(m.shape)
    rem_idx = np.argmax(np.nansum(squared_matrix, axis=0))
    squared_matrix.mask[rem_idx, :] = True
    squared_matrix.mask[:, rem_idx] = True
    return rem_idx


def get_score_function(function: str) -> Callable:
    score_functions = \
      {'min_mean_col': min_mean_col,
       'max_mean_col': max_mean_col,
       'min_inverse_rank_col_pref_large': min_inverse_rank_col_pref_large,
       'max_inverse_rank_col_pref_large': max_inverse_rank_col_pref_large,
       'min_inverse_rank_col_pref_small': min_inverse_rank_col_pref_small,
       'max_inverse_rank_col_pref_small': max_inverse_rank_col_pref_small,
       'max_weighted_dist': max_weighted_dist,
       'max_square_sum': max_square_sum}

    if function not in score_functions:
        print(f'Error: Undefined score function: :{function}', file=sys.stderr)
        return None

    return score_functions[function]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('score_function')
    parser.add_argument('-m', '--mask-value', type=float,
                        help='mask (ignore) value in input_file')
    parser.add_argument('-d', '--data-type',
                        help='specify data type of matrix (default: float)')
    args = parser.parse_args()

    input_file = args.input_file
    if not input_file.endswith(DATA_SUFFIX):
        print(f'Error: Expected {DATA_SUFFIX} file.', file=sys.stderr)
        sys.exit(1)

    if not args.data_type or args.data_type == 'float':
        data_type = None
    elif args.data_type == 'int':
        data_type = int
    else:
        print(f'Invalid data type specified: {args.data_type}', file=sys.stderr)
        sys.exit(1)

    score = get_score_function(args.score_function)
    if score is None:
        sys.exit(1)

    output_file = args.output_file

    headers, data = read_input(input_file, dtype=data_type)
    selector = Selector(headers,
                        data,
                        score=score,
                        summary=np.nanmean,
                        mask_value=args.mask_value)
    selector.process()
    with gzip.open(output_file, 'wt') as o:
        for line in selector.steps:
            o.write(DATA_DELIMITER.join(map(str, line)) + '\n')


if __name__ == '__main__':
    main()
    sys.exit(0)
