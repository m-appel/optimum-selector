import argparse
import bz2
import pickle
import sys
from collections import Counter

import numpy as np


OUTPUT_DELIMITER = ','


def read_input(input_file: str) -> dict:
    print(f'Reading date from: {input_file}')
    with bz2.open(input_file, 'r') as f:
        ret = pickle.load(f)
    if not isinstance(ret, dict):
        print(f'Error: Expected pickled dict, got: {type(ret)}',
              file=sys.stderr)
        return dict()
    print(f'Read {len(ret)} entries')
    l1_key = list(ret.keys())[0]
    l2_key = list(ret[l1_key].keys())[0]
    if type(ret[l1_key][l2_key]) != int:
        print(f'Casting data to int')
        ret = {l1: {l2: int(np.ceil(ret[l1][l2]))
                    for l2 in ret[l1]}
               for l1 in ret}
    return ret


def read_as_names(input_file: str) -> dict:
    print(f'Reading AS names from: {input_file}')
    ret = dict()
    with open(input_file, 'r') as f:
        for line in f:
            asn, name = line.strip().split(maxsplit=1)
            ret[int(asn)] = name
    print(f'Read {len(ret)} AS names')
    return ret


def weighted_dist(samples: dict) -> float:
    value_dist = Counter(samples.values())
    if 0 in value_dist:
        # Can happen for some reason in RTT.
        value_dist.pop(0)
    total_value_count = sum(value_dist.values())
    return sum([1 / value * count / total_value_count
                for value, count in value_dist.items()])


def filter_data(data: dict, min_samples: int) -> dict:
    len_pre = len(data)
    print(f'Removing entries with less than {min_samples} samples')
    data = {asn: samples
            for asn, samples in data.items()
            if len(samples) >= min_samples}
    len_post = len(data)
    print(f'Removed {len_pre - len_post} entries. Remaining: {len_post}')
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        help='.pickle.bz2 file containing distance vectors')
    parser.add_argument('output_file',
                        help='CSV file to which results are written')
    parser.add_argument('score_function',
                        help='scoring function used to rank candidates')
    parser.add_argument('--min-samples', type=int, default=0,
                        help='minimum number of samples required to include '
                             'an AS as a candidate (default: 0 -> include all '
                             'ASes)')
    parser.add_argument('--as-names',
                        help='optional list of AS names for pretty printing '
                             'top candidates')
    args = parser.parse_args()

    as_names_file = args.as_names
    as_names = dict()
    if as_names_file:
        as_names = read_as_names(as_names_file)

    input_file = args.input_file
    data = read_input(input_file)
    min_samples = args.min_samples
    if min_samples > 0:
        data = filter_data(data, min_samples)

    scores = [(asn, weighted_dist(samples)) for asn, samples in data.items()]
    scores.sort(key=lambda t: t[1])
    print('10 BEST CANDIDATES')
    for asn, score in scores[:10]:
        samples = list(data[asn].values())
        sample_count = len(samples)
        min_sample = min(samples)
        avg_sample = np.mean(samples)
        median_sample = np.median(samples)
        max_sample = max(samples)
        print(f'AS{asn}')
        print(f'Score: {score}')
        print(f'Samples: {sample_count} ({min_sample} / {avg_sample:.2f} / '
              f'{median_sample} / {max_sample})')
        print(f'Name: ', end='')
        if asn in as_names:
            print(as_names[asn], end='')
        print('\n')

    print('10 WORST CANDIDATES')
    for asn, score in scores[-10:]:
        samples = list(data[asn].values())
        sample_count = len(samples)
        min_sample = min(samples)
        avg_sample = np.mean(samples)
        median_sample = np.median(samples)
        max_sample = max(samples)
        print(f'AS{asn}')
        print(f'Score: {score}')
        print(f'Samples: {sample_count} ({min_sample} / {avg_sample:.2f} / '
              f'{median_sample} / {max_sample})')
        print(f'Name: ', end='')
        if asn in as_names:
            print(as_names[asn], end='')
        print('\n')

    output_file = args.output_file
    print(f'Writing {len(scores)} candidates to file: {output_file}')
    with open(output_file, 'w') as f:
        headers = ('asn', 'score', 'sample_count', 'min_sample', 'avg_sample',
                   'median_sample', 'max_sample')
        f.write(OUTPUT_DELIMITER.join(headers) + '\n')
        for asn, score in scores:
            samples = list(data[asn].values())
            sample_count = len(samples)
            min_sample = min(samples)
            avg_sample = np.mean(samples)
            median_sample = np.median(samples)
            max_sample = max(samples)
            f.write(OUTPUT_DELIMITER.join(map(str, (asn, score, sample_count,
                                                    min_sample, avg_sample,
                                                    median_sample, max_sample)
                                             )
                                         ) + '\n')



if __name__ == '__main__':
    main()
    sys.exit(0)
