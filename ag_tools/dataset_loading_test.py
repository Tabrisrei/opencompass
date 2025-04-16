import csv
import pdb
import json
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

@staticmethod
def load(path: str, name: str, **kwargs):
    path = get_data_path(path)
    # label_map = {
    #     1: 'A',
    #     2: 'B',
    #     3: 'C',
    #     4: 'D',
    # }

    dataset = DatasetDict()
    # for split in ['dev', 'train', 'test']:
    # for split in ['test', 'dev']:
    raw_data = []
    # if split == 'dev':
    #     filename = osp.join(path, 'JMED.jsonl')
    # elif split == 'test':
        # loading parquet file
    filename = osp.join(path, 'test_hard-00000-of-00001.parquet')
    print(f'Loading {filename}')
    # pdb.set_trace()
    dataset = load_dataset('parquet', data_files=filename, split='train')
    pdb.set_trace()
    # print(f'Loading {filename}')
    with open(filename, encoding='utf-8') as f:
        line = json.loads(f.readline())
        pdb.set_trace()
        # lines = json.load(f)
        for line in line:
            # print(f"line: {line}")
        # for line in f:
        #     line = json.loads(line)
            raw_data.append({
                'input': line['question'],
                'A': line['options']['A'],
                'B': line['options']['B'],
                'C': line['options']['C'],
                'target': line['answer'][0],
            })
        dataset = Dataset.from_list(raw_data)
    return dataset


if __name__ == '__main__':
    # path = '/home/gsb/opencompass/adatasets/temp/super-dainiu/medagents-benchmark/AfrimedQA'
    path = '/home/gsb/opencompass/adatasets/temp/super-dainiu/medagents-benchmark/MedQA'
    name = 'medbullets_op5'
    dataset = load(path, name)
    print(dataset)