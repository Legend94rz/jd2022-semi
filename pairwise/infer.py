from prj_config import Path, WORK_DATA, INPUT_DATA, PROJECT, PREPROCESS_OUTPUT, PREPROCESS_MOUNT, TEST_DATA, TRAIN_DATA, WEIGHT_OUTPUT, SUBMISSION_OUTPUT

import torch as T
import json
from collections import defaultdict
from typing import Dict, List
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from fasttorch import T, nn, F, Learner, SparseCategoricalAccuracy, EarlyStopping, ModelCheckpoint, F1Score, BinaryAccuracyWithLogits, TorchProfile
from fasttorch.misc.misc import seed_everything
from tqdm import tqdm
import datetime as dt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from scipy.special import softmax, expit as sigmoid
import lmdb
from pathlib import Path
import pathlib
import pickle as pkl
import random
import numpy  as np
from transformers import AutoTokenizer, AutoModel
import re
import argparse
from tools import get_all_keyattr, read_label_data, ufset, write_submit, extract_prop_from_title, std_obj
from defs import MoEx1d, YZBertTextEnc
from .train import TitleImg
seed_everything(43)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-w', '--weight_files', type=str, nargs='+', required=True)
    parser.add_argument('-o', '--output_file', type=str, required=True)
    return parser.parse_args()


class PairwiseTestDataset(Dataset):
    tokenizer = AutoTokenizer.from_pretrained("youzanai/bert-product-title-chinese", local_files_only=True)

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        a = self.data[index]
        return a['feature'], a['txt']

    def __len__(self):
        return len(self.data)

    @classmethod
    def collate_fn(cls, data):
        feature = T.tensor([x[0] for x in data], dtype=T.float32)
        inputs = cls.tokenizer([x[1] for x in data], return_tensors='pt', padding=True, truncation=True)
        return feature, inputs.input_ids, inputs.attention_mask


# pairwise-no-extra-neg-swa
if __name__ == "__main__":
    args = parse_args()
    with open(args.input_file, 'r') as fin:
        online_test_obj = [json.loads(x) for x in fin.readlines()]
    samples = []
    for x in online_test_obj:
        x = std_obj(x)
        samples.extend([{'img_name': x['img_name'], 'feature': x['feature'], 'query': a, 'txt': x['title'] if a == '图文' else  x['key_attr'][a]} for a in x['query']]) # 排除图文项
    ds = PairwiseTestDataset(samples)

    probs = []
    for i, pt in enumerate(args.weight_files):
        pt = Path(pt)
        if pt.exists():
            m = Learner(TitleImg(6), amp=True).load(pt, 'cuda')
            output = m.predict(ds, 256, device='cuda', collate_fn=PairwiseTestDataset.collate_fn)
            output = sigmoid(output).reshape(-1)
            probs.append(output)
        else:
            print(f"WARN: cannot find weights: {pt}")
    output = np.mean(probs, 0)
    
    submit = defaultdict(dict)
    assert len(samples) == len(output)
    for i in range(len(samples)):
        submit[samples[i]['img_name']][samples[i]['query']] = float(output[i])
    submit = [{'img_name': k, 'match': v} for k, v in submit.items()]
    write_submit(submit, args.output_file)
