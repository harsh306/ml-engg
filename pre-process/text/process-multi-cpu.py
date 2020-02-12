"""
Generate preprocesed data in parallel.

date: Nov 2019
author: harsh
"""

import argparse
import multiprocessing as mp
import re
from typing import Tuple, List
import pandas as pd

def read_test_file(filepath) -> Tuple[List, List, List]:
    """
    read test file and return list of source and target values
    We can even read in parallel.....
    :param filepath: path to test file
    :return: queries, ids
    """
    queries = []
    ids = []
    with open(filepath, encoding="utf-8") as infile:
        tmp = infile.readline()
        del tmp
        for line in infile:
            try:
                query, idx = line.split(",")
                queries.append(query)
                ids.append(int(idx))
            except ValueError:
                exit(1)
        assert len(queries) == len(ids)
        return queries, ids

def get_features(query: str, idx: int):
    """
    :param query: query to be processed
    :param idx: name identifier
    :return: None
    """
    query = re.sub('\W+', ' ', query)
    query = query.lower()
    data = {'query': query, 'idx': idx}

    export_frame = pd.DataFrame(data)
    export_frame.to_csv(f_exp, index=False, header=False, mode='a')


def export(test_file: str, export_path: str, multi_thread: bool):
    """
    export preprocessed features for a given a query
    :param test_file: path to file with test data
    :param export_path: path to output file
    :param multi_thread: yes or no
    :return: None
    """
    global f_exp
    f_exp = export_path
    # load test data from file
    queries, ids = read_test_file(test_file)

    if multi_thread:
        print(f"number of CPUs {mp.cpu_count()}")
        pool = mp.Pool()
        pool.starmap(get_features, zip(queries, ids))
    else:
        for query, idx in zip(queries, ids):
            get_features(query, idx)

    print('Exported...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="python process-multi-cpu.py test_file export_file")
    parser.add_argument("test", help="path to file with test data")
    parser.add_argument("export_path", help="path to export processed data")
    parser.add_argument("--multi-thread", action='store_true', default=False, help="multithread")
    args = parser.parse_args()

    result = export(test_file=args.test,
                    export_path=args.export_path,
                    multi_thread=args.multi_thread)
