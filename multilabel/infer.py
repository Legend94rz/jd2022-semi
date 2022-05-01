from prj_config import Path, WORK_DATA, INPUT_DATA, PROJECT, PREPROCESS_OUTPUT, PREPROCESS_MOUNT, TEST_DATA, TRAIN_DATA, WEIGHT_OUTPUT, SUBMISSION_OUTPUT

import torch as T
import json
from collections import defaultdict
from itertools import chain
from typing import Dict, List
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from fasttorch import T, nn, F, Learner, SparseCategoricalAccuracy, StochasticWeightAveraging, EarlyStopping, ModelCheckpoint, F1Score, BinaryAccuracyWithLogits, TorchProfile, Stage, LambdaLayer, CosineAnnealingWarmRestarts
from fasttorch.metrics.metrics import BaseMeter
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
import multiprocessing as mp
import re
from functools import partial
import argparse
from defs import LmdbObj, MoEx1d, PrjImg, OnlineMeter, VisualBert
from tools import ufset, std_obj, write_submit, get_eq_attr
seed_everything(43)


class MultiLabelTestDataset(Dataset):
    tokenizer = AutoTokenizer.from_pretrained("youzanai/bert-product-title-chinese", local_files_only=True)

    def __init__(self, objs):
        self.objs = objs

    def __getitem__(self, index):
        a = self.objs[index]
        return a['feature'], a['title']

    def __len__(self):
        return len(self.objs)

    @classmethod
    def collate_fn(cls, data):
        feature = T.tensor([x[0] for x in data], dtype=T.float32)
        inputs = cls.tokenizer([x[1] for x in data], return_tensors='pt', padding=True, truncation=True)
        return feature, inputs.input_ids, inputs.attention_mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-w', '--weight_files', type=str, nargs='+', required=True)
    parser.add_argument('-o', '--output_file', type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prop = get_eq_attr()
    with open(args.input_file, 'r') as fin:
        online_test_obj = [json.loads(x) for x in fin.readlines()]
    samples = []
    for x in online_test_obj:
        samples.append(std_obj(x))
    ds = MultiLabelTestDataset(samples)

    probs = []
    # 注意文件路径。注意参数
    for i, pt in enumerate(args.weight_files):
        pt = Path(pt)
        if pt.exists():
            m = Learner(VisualBert(len(prop) + 1, 6), amp=True).load(pt, 'cuda')
            output = m.predict(ds, 256, device='cuda', collate_fn=MultiLabelTestDataset.collate_fn)
            output = sigmoid(output)
            probs.append(output)
        else:
            print(f"WARN: cannot find weights: {pt}")
    output = np.mean(probs, 0)
    
    submit = []
    assert len(samples) == len(output)
    for i in range(len(samples)):
        row = {
            'img_name': samples[i]['img_name'],
            'match': {k: float(output[i][j]) for j, k in enumerate(prop) if k in samples[i]['query']}
        }
        if '图文' in samples[i]['query']:
            row['match']['图文'] = float(output[i][-1])
        submit.append(row)
    write_submit(submit, args.output_file)
