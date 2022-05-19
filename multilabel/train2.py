from prj_config import Path, WORK_DATA, INPUT_DATA, PROJECT, PREPROCESS_OUTPUT, PREPROCESS_MOUNT, TEST_DATA, TRAIN_DATA, WEIGHT_OUTPUT, SUBMISSION_OUTPUT

import torch as T
import json
from collections import defaultdict
from itertools import chain
from typing import Dict, List
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from fasttorch import T, nn, F, Learner, SparseCategoricalAccuracy, StochasticWeightAveraging, EarlyStopping, ModelCheckpoint, F1Score, BinaryAccuracyWithLogits, TorchProfile, Stage, LambdaLayer, CosineAnnealingWarmRestarts, ReduceLROnPlateau
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
from transformers import AutoTokenizer, AutoModel, LxmertModel
import multiprocessing as mp
import re
from functools import partial
from defs import LmdbObj, MoEx1d, PrjImg, OnlineMeter, random_delete, random_modify, random_replace, mutex_transform, sequential_transform, delete_words, replace_hidden, shuffle_title, get_lxmert
from tools import ufset, extract_color, extract_type, is_too_close_negtive
from .train import MultiLabelDataset, MultiLabelLearner
seed_everything(43)


class LXMERT(nn.Module):
    def __init__(self, ncls, nlayer=None):
        super().__init__()
        self.img_prj = nn.Sequential(MoEx1d(0.1))
        self.lxmert = get_lxmert("hfl/chinese-roberta-wwm-ext", nlayer)
        self.cls = nn.Linear(768, ncls)

    def forward(self, img, tt_id, tt_atmask):
        img = self.img_prj(img).unsqueeze(1)
        pos_shape = img.shape[:-1]
        o = self.lxmert(
            input_ids=tt_id, attention_mask=tt_atmask, 
            visual_feats=img, visual_pos=T.ones((img.shape[0], 1, 4), device=img.device),
        )
        return self.cls(o.pooled_output)


def train_lxmert(fd):
    feature_db = LmdbObj(PREPROCESS_MOUNT / 'feature_db', 'train')
    #train_obj = LmdbObj(PREPROCESS_MOUNT / f'full.json', 'train')
    train_obj = LmdbObj(PREPROCESS_MOUNT / f'fold-{fd}.json', 'train')
    val_obj = LmdbObj(PREPROCESS_MOUNT / f'fold-{fd}.json', 'val')
    EPOCHS = 1000

    trans = mutex_transform([
        sequential_transform([
            random_replace(candidate_attr, candidate_title),
            random_delete(0.3),
            random_modify(0.3, prop)
        ], [0.5, 0.4, 0.5]),
        mutex_transform([
            delete_words(), # 分词后随机选择至少一个短语，删除。相应修改 match 的字段。
            replace_hidden(rep_color=False, rep_tp=True),  # 随机换类型、颜色或性别中的至少一个，没有这些则保持原输入。
            sequential_transform([
                delete_words(), # 分词后随机选择至少一个短语，删除。相应修改 match 的字段。
                replace_hidden(rep_color=False, rep_tp=True),  # 随机换类型、颜色或性别中的至少一个，没有这些则保持原输入。
            ], [1., 1.])
        ], [0.33, 0.33, 0.34]),
        lambda x: x
    ], [0.5, 0.45, 0.05])
    model = LXMERT(len(prop)+1, (4, 3, 3))
    #opt = T.optim.AdamW([
    #    {'params': [p for n, p in model.named_parameters() if n.startswith('lxmert.embeddings.') or n.startswith('lxmert.encoder.layer')], 'lr': 2e-5, 'weight_decay': 1e-6},
    #    {'params': [p for n, p in model.named_parameters() if not(n.startswith('lxmert.embeddings.') or n.startswith('lxmert.encoder.layer'))], 'lr': 1e-4},
    #])
    opt = T.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-6)
    m = MultiLabelLearner(model, opt, [nn.BCEWithLogitsLoss()], amp=True)
    _, val_log = m.fit(MultiLabelDataset(train_obj, feature_db, prop, trans, 2), EPOCHS, 200, [(0, 'online', OnlineMeter(len(prop_ncls)))], 
        validation_set=MultiLabelDataset(val_obj, feature_db, prop),
        callbacks=[
            #StochasticWeightAveraging(1e-6, str(WEIGHT_OUTPUT / f'lxmert-swa-full.pt'), 16, anneal_epochs=8), 
            ReduceLROnPlateau(factor=0.5, patience=1, threshold=0.002, verbose=True),
            EarlyStopping('val_online', 5, 'max'), 
            #ModelCheckpoint(str(WEIGHT_OUTPUT / f'lxmert-{fd}.pt'), 'val_online', mode='max')
        ], 
        device=f'cuda:{fd}', collate_fn=MultiLabelDataset.collate_fn, num_workers=8, shuffle=True
    )
    print(val_log['val_online'].max())


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

    #train_lxmert(0)
    procs = []
    for i in range(5):
        p = mp.Process(target=train_lxmert, args=(i, ))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        p.close()