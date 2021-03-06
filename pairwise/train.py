from prj_config import Path, WORK_DATA, INPUT_DATA, PROJECT, PREPROCESS_OUTPUT, PREPROCESS_MOUNT, TEST_DATA, TRAIN_DATA, WEIGHT_OUTPUT, SUBMISSION_OUTPUT

import torch as T
import json
from collections import defaultdict
from itertools import chain
from typing import Dict, List
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from fasttorch import T, nn, F, Learner, SparseCategoricalAccuracy, StochasticWeightAveraging, EarlyStopping, ModelCheckpoint, F1Score, BinaryAccuracyWithLogits, TorchProfile, Stage, LambdaLayer, CosineAnnealingWarmRestarts, ReduceLROnPlateau, LmdbDict
from fasttorch.metrics.metrics import BaseMeter
from fasttorch.misc.misc import seed_everything
from tqdm import tqdm
import datetime as dt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from scipy.special import softmax
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
from defs import MoEx1d
from tools import ufset
seed_everything(43)


class YZBertTextEnc(nn.Module):
    def __init__(self, nlayer=None):
        super().__init__()
        self.enc = AutoModel.from_pretrained("youzanai/bert-product-title-chinese", local_files_only=True).eval().requires_grad_(False)
        if nlayer is not None:
            self.enc.encoder.layer = self.enc.encoder.layer[:nlayer]

    def forward(self, input_ids, attention_mask):
        return self.enc(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False).last_hidden_state
    

class TitleImg(nn.Module):
    def __init__(self, nlayer=None):
        super().__init__()
        d = 128
        self.img_prj = nn.Sequential(nn.Dropout(0.15), nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, d))
        self.txt_enc = YZBertTextEnc(nlayer)
        self.txt_prj = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, d))
        self.padmask = nn.ConstantPad1d((1, 0), 1)
        enc = nn.TransformerEncoderLayer(d, 4, dim_feedforward=256, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, 5)
        self.overall = nn.Sequential(nn.Dropout(0.15), nn.Linear(d, 1))  # ??????
    
    def forward(self, img, tt_id, tt_atmask):
        img = self.img_prj(img).unsqueeze(1) # [B, 1, d]
        txt = self.txt_prj(self.txt_enc(tt_id, tt_atmask))  # [B, L, 768] => [B, L, d]
        #import ipdb; ipdb.set_trace()
        x = T.cat([img, txt], dim=1)
        atmask = self.padmask(tt_atmask)
        x = self.transformer(x, src_key_padding_mask=atmask)  # [B, 1+L, d]
        overall = self.overall(x[:, 0, :])
        return overall


class PairwiseDataset(Dataset):
    tokenizer = AutoTokenizer.from_pretrained("youzanai/bert-product-title-chinese", local_files_only=True)

    def __init__(self, data, feature_db):
        #data: [{img_name, txt, match}, ...]
        self.data = data
        self.feature_db = feature_db

    def __getitem__(self, index):
        a = self.data[index]
        return self.feature_db[a['img_name']], a['txt'], a['match']

    def __len__(self):
        return len(self.data)

    @classmethod
    def collate_fn(cls, data):
        feature = T.tensor([x[0] for x in data], dtype=T.float32)
        inputs = cls.tokenizer([x[1] for x in data], return_tensors='pt', padding=True, truncation=True)
        label = T.tensor([x[2] for x in data], dtype=T.float32).reshape(-1, 1)
        return feature, inputs.input_ids, inputs.attention_mask, label


def make_sample(obj, prop, ufset):
    # ??????????????????
    # ??????????????????????????????
    #  *???????????????????????????1??????????????????????????????????????????????????????
    res = []
    attr = obj['key_attr']
    for k, v in attr.items():
        if k in {'??????', '??????'}:
            continue
        #if k == '??????':
        #    values = np.random.choice(prop[k], len(prop[k]) // 2, replace=False)
        #else:
        values = prop[k]
        for x in values:
            res.append({'img_name': obj['img_name'], 'txt': x, 'match': int(ufset[v] == ufset[x])})
    #others = list(set(prop) - set(attr))
    #k = np.random.choice(others)
    #for x in prop[k]:
    #    res.append({'img_name': obj['img_name'], 'txt': x, 'match': 0})
    return res


def pos_aug(obj, ufset):
    # ???????????????obj ?????????????????????????????????
    # ?????????????????????????????????(????????????5???)
    # ????????????????????????
    res = [{'img_name': obj['img_name'], 'txt': obj['title'], 'match': 1}]
    for k, v in obj['key_attr'].items():
        ntitle = obj['title']
        for x in set(obj['key_attr']) - {k}:
            ntitle = ntitle.replace(obj['key_attr'][x], '')
        res.append({'img_name': obj['img_name'], 'txt': ntitle, 'match': 1})
    ntitle = obj['title']
    for k, v in obj['key_attr'].items():
        ntitle = ntitle.replace(v, '')
    res.append({'img_name': obj['img_name'], 'txt': ntitle, 'match': 1})
    return res
    
    
def neg_aug(obj, all_prop, ufset, table: pd.DataFrame, grouped_df: pd.DataFrame):
    # ???????????????obj ?????????????????????????????????
    # ?????????????????????????????????????????????????????? 4 ??????
    # ??????
    # ??????????????????????????????
    assert obj['match']['??????'], f"Cannot apply neg_aug to an unmatched object: {obj['img_name']} {obj['key_attr']} {obj['match']}"
    res = []
    for k, v in obj['key_attr'].items():
        s = list({x for x in all_prop[k] if ufset[x] != ufset[v]})
        nv = np.random.choice(s)
        res.append({'img_name': obj['img_name'], 'txt': obj['title'].replace(v, nv), 'match': 0})
    n = len(res)
    for _ in range(n):
        i = np.random.choice(len(table))
        row = table.iloc[i]
        res.append({'img_name': obj['img_name'], 'txt': row['title'], 'match': 0})
    #query = [[k, v] for k, v in obj['key_attr'].items()]
    #if query:
    #    query = list(zip(*query))
    #    tmp = grouped_df.xs(query[1], level=query[0])
    #    if len(tmp)>0:
    #        newtitles = list(tmp.iloc[np.random.choice(len(tmp))])
    #        for t in np.random.choice(newtitles, min(len(newtitles), n), replace=False):
    #            if not set.issubset(set(t) ^ set(obj['title']), {'???', '???', '???', '???', '???'}):
    #                res.append({'img_name': obj['img_name'], 'txt': t, 'match': 0})
    return res


def train_pairwise(fd):
    feature_db = LmdbDict(PREPROCESS_MOUNT / 'feature_db', 'train', lock=False)

    train_obj = LmdbDict(PREPROCESS_MOUNT / f'fold-{fd}.json', 'train', lock=False)
    unmatched_train = LmdbDict(PREPROCESS_MOUNT / f'fold-{fd}.json', 'unmatched_train', lock=False)
    samples = []
    for x in tqdm(train_obj.values()):
        samples.extend(make_sample(x, prop, ufset))
        samples.extend(neg_aug(x, prop, ufset, table, grouped_df))
        samples.extend(pos_aug(x, ufset))
    for x in tqdm(unmatched_train.values()):
        samples.append({'img_name': x['img_name'], 'txt': x['title'], 'match': 0})
    print(sum([x['match'] for x in samples]), len(samples))
    
    val_samples = []
    unique = set()
    val_obj = LmdbDict(PREPROCESS_MOUNT / f'fold-{fd}.json', 'val', lock=False)
    unmatched_val = LmdbDict(PREPROCESS_MOUNT / f'fold-{fd}.json', 'unmatched_val', lock=False)
    for x in tqdm(val_obj.values()):
        if x['img_name'] not in unique:
            x = {'img_name': x['img_name'], 'key_attr': x['gt_key_attr'], 'match': {**{'??????': 1}, **{k:1 for k in x['gt_key_attr']}}, 'title': x['gt_title'], 'gt_key_attr': x['gt_key_attr'].copy()}
            val_samples.extend(make_sample(x, prop, ufset))
            val_samples.extend(neg_aug(x, prop, ufset, table, grouped_df))
            val_samples.extend(pos_aug(x, ufset))
            unique.add(x['img_name'])
    for x in tqdm(unmatched_val.values()):
        val_samples.append({'img_name': x['img_name'], 'txt': x['title'], 'match': 0})
    print(sum([x['match'] for x in val_samples]), len(val_samples))

    m = Learner(TitleImg(6), T.optim.Adam, [nn.BCEWithLogitsLoss()], amp=True)
    _, val_log = m.fit(PairwiseDataset(samples, feature_db), 1000, 256, [(0, 'acc', BinaryAccuracyWithLogits())], 
        validation_set=PairwiseDataset(val_samples, feature_db),
        callbacks=[
            StochasticWeightAveraging(1e-4, str(WEIGHT_OUTPUT / f'pairwise-no-extra-neg-swa-{fd}.pt'), 5),
            EarlyStopping('val_loss', 4, 'min'), 
            ModelCheckpoint(str(WEIGHT_OUTPUT / f'pairwise-no-extra-neg-{fd}.pt'), 'val_loss', mode='min')
        ], 
        device=f'cuda:{fd}', collate_fn=PairwiseDataset.collate_fn, num_workers=8, shuffle=True
    )
    print(val_log['val_acc'].max())


if __name__ == "__main__":
    with open(PREPROCESS_MOUNT / 'props.pkl', 'rb') as fin:
        data = pkl.load(fin)
        prop = data['prop']
        all_prop = data['all_prop']
        prop2id = data['prop2id']
        prop_ncls = data['prop_ncls']
        literals = data['literals']
        literal_embedding_bert = data['literal_embedding_bert']
        literal2id = data['literal2id']
        candidate_title = data['candidate_title']
        candidate_attr = data['candidate_attr']
        candidate_meta = data['candidate_meta']
    table = pd.concat([pd.DataFrame(candidate_title, columns=['title']), pd.DataFrame.from_records(candidate_attr)], axis=1)
    grouped_df = table.groupby(list(prop), dropna=False)['title'].agg(set)

    #train_pairwise(0)
    procs = []
    for i in range(5):
        p = mp.Process(target=train_pairwise, args=(i, ))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        p.close()
