import argparse
import gzip
import logging
import json
import sys
from typing import Tuple
from random import shuffle

import numpy as np
import requests
from requests_futures.sessions import FuturesSession


API_BASE = 'https://atlas.ripe.net/api/v2/'
DELIMITER = ','


def process_response(response: requests.Request, *args, **kwargs) -> None:
    if response.status_code != requests.codes.ok:
        logging.error(f'Request to {response.url} failed with status: '
                      f'{response.status_code}')
        response.data =  None
        return
    try:
        data = response.json()
    except json.decoder.JSONDecodeError as e:
        logging.error(f'Decoding JSON reply from {response.url} failed with '
                      f'exception: {e}')
        response.data =  None
        return
    response.data = data


def get_probe_per_as(asns: list,
                     num_probes: int,
                     randomize: bool) -> Tuple[list, int]:
    if num_probes > 500:
        logging.error(f'More than 500 probes per AS not yet supported.')
        return list(), -1

    # We can make the requests a bit more efficient if we do not
    # randomize, by only requesting the exact number of probes
    # required.
    if randomize:
        logging.info(f'Randomizing probe selection.')
        params = {'format': 'json', 'page_size': 500, 'status': 1}
    else:
        params = {'format': 'json', 'page_size': num_probes, 'status': 1}

    session = FuturesSession()
    url = f'{API_BASE}probes'

    queries = list()
    for asn in asns:
        req_params = params.copy()
        req_params['asn_v4'] = asn
        queries.append((session.get(url,
                                    params=req_params,
                                    hooks={'response': process_response}),
                        asn))

    probes = list()
    as_with_probes = 0

    for query, asn in queries:
        response = query.result()
        if response.data is None:
            continue
        result = response.data
        if result['count'] == 0:
            logging.error(f'Found no connected probe for AS {asn}')
            continue
        as_with_probes += 1
        if result['count'] < num_probes:
            logging.warning(f'Failed to get requested number of probes '
                            f'({num_probes}) for AS {asn}. '
                            f'Got: {result["count"]}')
        asn_probe_ids = [res['id'] for res in result['results']]
        if randomize:
            shuffle(asn_probe_ids)
        probes += asn_probe_ids[:num_probes + 1]

    if as_with_probes != len(asns):
        logging.warning(f'Failed to get probes for all ASes. ASes: {len(asns)} '
                        f'with probes: {as_with_probes}')
    return probes, as_with_probes


def get_asns(steps_file: str) -> list:
    logging.info(f'Loading ASes from file: {steps_file}')
    ret = list()
    with gzip.open(steps_file, 'rt') as f:
        f.readline()
        for line in f:
            line_split = line.split(DELIMITER)
            if len(line_split) != 3:
                logging.error(f'Steps file has invalid line format: '
                              f'{line.strip()}')
                return list()
            if line_split[0] == '--':
                continue
            ret.append(int(line_split[1]))
    return ret


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('steps_file')
    parser.add_argument('as_count', type=int)
    parser.add_argument('output_file')
    parser.add_argument('-p', '--probe-count', type=int, default=1,
                        help='set the number of probes per AS')
    parser.add_argument('-r', '--randomize', action='store_true',
                        help='shuffle per-AS probe list before selecting '
                             'probes')
    args = parser.parse_args()

    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    asns = get_asns(args.steps_file)
    if not asns:
        sys.exit(1)

    as_count = args.as_count

    prb_ids = list()
    interval_end = len(asns)
    interval_start = interval_end - as_count
    as_with_probes = 0
    while as_with_probes < as_count:
        logging.info(f'Probe count: {len(prb_ids)} '
                     f'Interval: {interval_start} - {interval_end} '
                     f'(len: {interval_end - interval_start})')
        interval_prb_ids, interval_as_count = \
            get_probe_per_as(asns[interval_start:interval_end],
                             args.probe_count,
                             args.randomize)
        if not interval_prb_ids:
            sys.exit(1)

        prb_ids += interval_prb_ids
        as_with_probes += interval_as_count
        missing_probes = as_count - as_with_probes
        interval_end = interval_start
        interval_start = interval_start - missing_probes

        if interval_start < 0:
            logging.error(f'No more ASes left to try.')
            sys.exit(1)

    output_file = args.output_file
    logging.info(f'Writing {len(prb_ids)} probe IDs to file: {output_file}')

    with open(args.output_file, 'w') as f:
        f.write(DELIMITER.join(map(str, prb_ids)) + '\n')


if __name__ == '__main__':
    main()
    sys.exit(0)
