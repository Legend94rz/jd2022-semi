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
from transformers import AutoTokenizer, AutoModel
import multiprocessing as mp
import re
from functools import partial
from defs import LmdbObj, MoEx1d, PrjImg, OnlineMeter, VisualBert, random_delete, random_modify, random_replace, mutex_transform, sequential_transform, delete_words, replace_hidden, shuffle_title
from tools import ufset, extract_color, extract_type, is_too_close_negtive
from .train import MultiLabelDataset, MultiLabelLearner
seed_everything(43)


class BertTextEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained("bert-base-chinese", local_files_only=True)

    def forward(self, input_ids, attention_mask):
        return self.enc(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False).pooler_output


class ConcatBert(nn.Module):
    def __init__(self, ncls):
        super().__init__()
        d = 256
        self.txt = BertTextEnc()
        self.img = nn.Sequential(MoEx1d(0.1), nn.Linear(2048, 256), nn.Dropout(0.1), nn.ReLU(), nn.Linear(256, d))
        self.fc = nn.Sequential(nn.Linear(d+768, 512), nn.Dropout(0.1), nn.ReLU(), nn.Linear(512, ncls))

    def forward(self, img, tt_ids, tt_atmask):
        txt = self.txt(tt_ids, tt_atmask)
        img = self.img(img)
        o = self.fc(T.cat([txt, img], axis=-1))
        return o


def train_concat_bert(fd):
    feature_db = LmdbObj(PREPROCESS_MOUNT / 'feature_db', 'train')
    train_obj = LmdbObj(PREPROCESS_MOUNT / f'full.json', 'train')
    #train_obj = LmdbObj(PREPROCESS_MOUNT / f'fold-{fd}.json', 'train')
    #val_obj = LmdbObj(PREPROCESS_MOUNT / f'fold-{fd}.json', 'val')
    EPOCHS = 30

    trans = sequential_transform([
            random_replace(candidate_attr, candidate_title),
            random_delete(0.3),
            random_modify(0.3, prop)
        ], [0.4, 0.4, 0.4])

    m = MultiLabelLearner(ConcatBert(len(prop)+1), partial(T.optim.AdamW, lr=2e-5, weight_decay=1e-6), [nn.BCEWithLogitsLoss()], amp=True)
    _, val_log = m.fit(MultiLabelDataset(train_obj, feature_db, prop, trans, 5), EPOCHS, 256, [(0, 'online', OnlineMeter(len(prop_ncls)))], 
        #validation_set=MultiLabelDataset(val_obj, feature_db, prop),
        callbacks=[
            StochasticWeightAveraging(1e-6, str(WEIGHT_OUTPUT / f'concatbert-swa-full.pt'), 18, anneal_epochs=12), 
            #ReduceLROnPlateau(factor=0.5, patience=1, threshold=0.002, verbose=True),
            #EarlyStopping('val_online', 3, 'max'), 
            #ModelCheckpoint(str(WEIGHT_OUTPUT / f'concatbert-{fd}.pt'), 'val_online', mode='max')
        ], 
        # todo: device
        device=f'cuda:2', collate_fn=MultiLabelDataset.collate_fn, num_workers=8, shuffle=True
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

    train_concat_bert(0)
    #procs = []
    #for i in range(5):
    #    p = mp.Process(target=train_concat_bert, args=(i, ))
    #    p.start()
    #    procs.append(p)
    #for p in procs:
    #    p.join()
    #    p.close()