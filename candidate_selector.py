import argparse
import bz2
import logging
import pickle
import sys
from collections import Counter

import numpy as np


OUTPUT_DELIMITER = ','


def read_input(input_file: str) -> dict:
    logging.info(f'Reading data from: {input_file}')
    with bz2.open(input_file, 'r') as f:
        ret = pickle.load(f)
    if not isinstance(ret, dict):
        logging.info(f'Error: Expected pickled dict, got: {type(ret)}',
              file=sys.stderr)
        return dict()
    logging.info(f'Read {len(ret)} entries')
    l1_key = list(ret.keys())[0]
    l2_key = list(ret[l1_key].keys())[0]
    if type(ret[l1_key][l2_key]) != int:
        logging.info(f'Casting data to int')
        ret = {l1: {l2: int(np.ceil(ret[l1][l2]))
                    for l2 in ret[l1]}
               for l1 in ret}
    if 'as_hops' in input_file:
        logging.info('AS hops detected, increasing by 1 hop to produce AS-path '
              'length')
        ret = {l1: {l2: int(ret[l1][l2] + 1)
                    for l2 in ret[l1]}
               for l1 in ret}
    return ret


def read_as_prefixes(input_file: str) -> dict:
    logging.info(f'Reading AS prefix counts from: {input_file}')
    ret = dict()
    with open(input_file, 'r') as f:
        f.readline()
        for line in f:
            asn, prefix_count = line.strip().split(',')
            ret[int(asn)] = int(prefix_count)
    logging.info(f'Read {len(ret)} prefix counts')
    return ret


def read_as_names(input_file: str) -> dict:
    logging.info(f'Reading AS names from: {input_file}')
    ret = dict()
    with open(input_file, 'r') as f:
        for line in f:
            asn, name = line.strip().split(maxsplit=1)
            ret[int(asn)] = name
    logging.info(f'Read {len(ret)} AS names')
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
    logging.info(f'Removing entries with less than {min_samples} samples')
    data = {asn: samples
            for asn, samples in data.items()
            if len(samples) >= min_samples}
    len_post = len(data)
    logging.info(f'Removed {len_pre - len_post} entries. Remaining: {len_post}')
    return data


def calculate_scores(data: dict, min_samples: int) -> list:
    if min_samples > 0:
        data = filter_data(data, min_samples)

    scores = [(asn, weighted_dist(samples)) for asn, samples in data.items()]
    scores.sort(key=lambda t: t[1])
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        help='.pickle.bz2 file containing distance vectors')
    parser.add_argument('output_file',
                        help='CSV file to which results are written')
    parser.add_argument('as_prefix_file',
                        help='CSV file containing the AS to prefix count')
    parser.add_argument('--min-samples', type=int, default=0,
                        help='minimum number of samples required to include '
                             'an AS as a candidate (default: 0 -> include all '
                             'ASes)')
    parser.add_argument('--as-names',
                        help='optional list of AS names for pretty printing '
                             'top candidates')
    args = parser.parse_args()

    log_fmt = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(
        format=log_fmt,
        level=logging.INFO,
        filename='candidate_selector.log',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    as_names_file = args.as_names
    as_names = dict()
    if as_names_file:
        as_names = read_as_names(as_names_file)

    as_prefix_file = args.as_prefix_file
    as_prefix_counts = read_as_prefixes(as_prefix_file)

    input_file = args.input_file
    data = read_input(input_file)
    min_samples = args.min_samples

    scores = calculate_scores(data, min_samples)

    logging.info('10 BEST CANDIDATES')
    for asn, score in scores[:10]:
        samples = list(data[asn].values())
        prefix_count = None
        if asn in as_prefix_counts:
            prefix_count = as_prefix_counts[asn]
        sample_count = len(samples)
        min_sample = min(samples)
        avg_sample = np.mean(samples)
        median_sample = np.median(samples)
        max_sample = max(samples)
        logging.info(f'AS{asn}')
        logging.info(f'Score: {score}')
        logging.info(f'Prefixes: {prefix_count}')
        logging.info(f'Samples: {sample_count} ({min_sample} / '
                     f'{avg_sample:.2f} / {median_sample} / {max_sample})')
        if asn in as_names:
            logging.info(f'Name: {as_names[asn]}')

    logging.info('10 WORST CANDIDATES')
    for asn, score in scores[-10:]:
        samples = list(data[asn].values())
        prefix_count = None
        if asn in as_prefix_counts:
            prefix_count = as_prefix_counts[asn]
        sample_count = len(samples)
        min_sample = min(samples)
        avg_sample = np.mean(samples)
        median_sample = np.median(samples)
        max_sample = max(samples)
        logging.info(f'AS{asn}')
        logging.info(f'Score: {score}')
        logging.info(f'Prefixes: {prefix_count}')
        logging.info(f'Samples: {sample_count} ({min_sample} / '
                     f'{avg_sample:.2f} / {median_sample} / {max_sample})')
        if asn in as_names:
            logging.info(f'Name: {as_names[asn]}')

    output_file = args.output_file
    logging.info(f'Writing {len(scores)} candidates to file: {output_file}')
    with open(output_file, 'w') as f:
        headers = ('asn', 'score', 'prefix_count', 'sample_count',
                   'min_sample', 'avg_sample', 'median_sample', 'max_sample')
        f.write(OUTPUT_DELIMITER.join(headers) + '\n')
        for asn, score in scores:
            samples = list(data[asn].values())
            prefix_count = 0
            if asn in as_prefix_counts:
                prefix_count = as_prefix_counts[asn]
            else:
                logging.info(f'Warning: Found no prefix count for AS{asn}')
            sample_count = len(samples)
            min_sample = min(samples)
            avg_sample = np.mean(samples)
            median_sample = np.median(samples)
            max_sample = max(samples)
            f.write(OUTPUT_DELIMITER.join(map(str, (asn, score, prefix_count,
                                                    sample_count, min_sample,
                                                    avg_sample, median_sample,
                                                    max_sample)
                                             )
                                         ) + '\n')



if __name__ == '__main__':
    main()
    sys.exit(0)
