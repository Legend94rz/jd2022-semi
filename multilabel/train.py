from prj_config import Path, WORK_DATA, INPUT_DATA, PROJECT, PREPROCESS_OUTPUT, PREPROCESS_MOUNT, TEST_DATA, TRAIN_DATA, WEIGHT_OUTPUT, SUBMISSION_OUTPUT

import torch as T
import json
from collections import defaultdict
from itertools import chain
from typing import Dict, List
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from fasttorch import LmdbDict, T, nn, F, Learner, SparseCategoricalAccuracy, StochasticWeightAveraging, EarlyStopping, ModelCheckpoint, F1Score, BinaryAccuracyWithLogits, TorchProfile, Stage, LambdaLayer, CosineAnnealingWarmRestarts, ReduceLROnPlateau
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
from defs import LmdbObj, MoEx1d, OnlineMeter, VisualBert, random_delete, random_modify, random_replace, mutex_transform, sequential_transform, delete_words, replace_hidden, shuffle_title
from tools import ufset, extract_color, extract_type, is_too_close_negtive
seed_everything(43)
BERT = 'youzanai/bert-product-title-chinese'


class MultiLabelDataset(Dataset):
    tokenizer = AutoTokenizer.from_pretrained(BERT, local_files_only=True)

    def __init__(self, objs, feature_db, prop, trans=None, enlarge=1):
        self.objs = objs
        self.feature_db = feature_db
        self.prop = prop
        self.trans = trans
        self.enlarge = enlarge

    def __getitem__(self, index):
        a = self.objs[index % len(self.objs)]
        if self.trans:
            while True:
                a = self.trans(a)
                if not is_too_close_negtive(a):
                    break
        qmatch = [a['match'].get(k, 0) for k in self.prop] + [a['match']['??????']]  # ??????????????? [??????]
        qmask = [int(k in a['match']) for k in self.prop] + [1]
        return self.feature_db[a['img_name']], a['title'], qmask, qmatch  # qmask, for kl_div & metric; gtmask & gttarget, for img; 

    def __len__(self):
        return int(len(self.objs) * self.enlarge)

    @classmethod
    def collate_fn(cls, data):
        feature = T.tensor([x[0] for x in data], dtype=T.float32)
        inputs = cls.tokenizer([x[1] for x in data], return_tensors='pt', padding=True, truncation=True)
        qmask = T.tensor([x[2] for x in data], dtype=T.float32)
        label = T.tensor([x[3] for x in data], dtype=T.float32)
        return feature, inputs.input_ids, inputs.attention_mask, qmask, label


class MultiLabelLearner(Learner):
    def compute_forward(self, batch_data, stage):
        if stage == Stage.INFERENCE:
            return self.module(*batch_data)
        return self.module(*batch_data[:3])

    def compute_losses(self, forward_results, batch_data):
        logits = forward_results[0]
        *_, qmask, qmatch = batch_data
        pmatch = qmatch[:, :-1]
        pmask = qmask[:, :-1]
        overall = qmatch[:, -1].reshape(-1, 1)
        masked_min_match = ((1.0 - pmask) * pmatch.max().detach() + pmask * pmatch).min(1, keepdims=True).values
        reg = F.kl_div(F.logsigmoid(masked_min_match), F.logsigmoid(overall), reduction='batchmean', log_target=True)  # ??????overall?????????query?????????match??????==1?????????matchc????????? query==0????????? overall ???==0???
        loss = 0.7*F.binary_cross_entropy_with_logits(logits, qmatch) + 0.3*reg
        return loss

    def compute_metric(self, idx, name, func, detached_results, batch_data):
        if name == 'online':
            match = detached_results[0]
            *_, qmask, qmatch = batch_data
            func(match[:, :-1], match[:, -1:], qmask[:, :-1], qmatch[:, :-1], qmatch[:, -1:])
        else:
            super().compute_metric(idx, name, func, detached_results, batch_data)


def train_multilabel(fd):
    feature_db = LmdbDict(PREPROCESS_MOUNT / 'feature_db', 'train', lock=False)
    train_obj = LmdbDict(PREPROCESS_MOUNT / f'full.json', 'train', lock=False)
    #train_obj = LmdbDict(PREPROCESS_MOUNT / f'fold-{fd}.json', 'train', lock=False)
    #val_obj = LmdbDict(PREPROCESS_MOUNT / f'fold-{fd}.json', 'val', lock=False)
    EPOCHS = 26

    trans = mutex_transform([
        sequential_transform([
            random_replace(candidate_attr, candidate_title),  # ????????????????????????
            random_delete(0.3), # ????????????????????????
            random_modify(0.3, prop) # ????????????????????????
        ], [0.5, 0.4, 0.5]),
        mutex_transform([
            delete_words(), # ??????????????????????????????????????????????????????????????? match ????????????
            replace_hidden(rep_color=False, rep_tp=True),  # ???????????????????????????????????????????????????????????????????????????????????????
            sequential_transform([
                delete_words(),
                replace_hidden(rep_color=False, rep_tp=True),
            ], [1., 1.])
        ], [0.33, 0.33, 0.34]),
        lambda x: x
    ], [0.5, 0.45, 0.05])

    m = MultiLabelLearner(VisualBert(len(prop)+1, 6, BERT), partial(T.optim.AdamW, lr=2e-5, weight_decay=1e-6), [nn.BCEWithLogitsLoss()], amp=True)
    _, val_log = m.fit(MultiLabelDataset(train_obj, feature_db, prop, trans, 5), EPOCHS, 256, [(0, 'online', OnlineMeter(len(prop_ncls)))], 
        #validation_set=MultiLabelDataset(val_obj, feature_db, prop),
        callbacks=[
            StochasticWeightAveraging(1e-6, str(WEIGHT_OUTPUT / f'visualbert-swa-full.pt'), 16, anneal_epochs=10), 
            #ReduceLROnPlateau(factor=0.5, patience=1, threshold=0.002, verbose=True),
            #EarlyStopping('val_online', 3, 'max'), 
            #ModelCheckpoint(str(WEIGHT_OUTPUT / f'visualbert-{fd}.pt'), 'val_online', mode='max')
        ], 
        device=f'cuda:{fd}', collate_fn=MultiLabelDataset.collate_fn, num_workers=8, shuffle=True
    )
    #print(val_log['val_online'].max())


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
    table['??????'] = table['title'].map(extract_color)
    table['??????'] = table['title'].map(extract_type)
    grouped_df = table.groupby(list(prop)+['??????', '??????'], dropna=False)['title'].agg(lambda titles: list(set(titles))).reset_index()

    train_multilabel(0)
    #procs = []
    #for i in range(5):
    #    p = mp.Process(target=train_multilabel, args=(i, ))
    #    p.start()
    #    procs.append(p)
    #for p in procs:
    #    p.join()
    #    p.close()
