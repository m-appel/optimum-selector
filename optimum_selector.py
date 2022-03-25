import argparse
import gzip
import logging
import sys
from typing import Any, List, Tuple

import numpy as np

from selector import Selector
from score_functions import get_score_function


DATA_SUFFIX = '.csv.gz'
DATA_DELIMITER = ','


def read_input(input_file: str, dtype = None) -> Tuple[List[Any], np.ndarray]:
    with gzip.open(input_file, 'rt') as f:
        headers = f.readline().lstrip('#').strip().split(',')
    data = np.loadtxt(input_file, delimiter=DATA_DELIMITER)
    if dtype:
        return headers, np.asarray(data, dtype=dtype)
    return headers, data


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

    log_fmt = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(
        format=log_fmt,
        level=logging.INFO,
        filename='optimum_selector.log',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    input_file = args.input_file
    if not input_file.endswith(DATA_SUFFIX):
        logging.error(f'Expected {DATA_SUFFIX} file.')
        sys.exit(1)

    if not args.data_type or args.data_type == 'float':
        data_type = None
    elif args.data_type == 'int':
        data_type = int
    else:
        logging.error(f'Invalid data type specified: {args.data_type}')
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
