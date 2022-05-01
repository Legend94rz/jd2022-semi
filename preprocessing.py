from prj_config import Path, WORK_DATA, INPUT_DATA, PROJECT, PREPROCESS_OUTPUT, TEST_DATA, TRAIN_DATA, WEIGHT_OUTPUT, SUBMISSION_OUTPUT

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from tools import get_all_keyattr, extract_prop_from_title, get_eq_attr, std_obj, get_meta, hash_obj, is_too_close_negtive
from defs import random_delete, random_modify, random_replace, sequential_transform, mutex_transform, YZBertTextEnc, YZCLIPTextEnc
import torch as T
from itertools import chain
from functools import partial
import pickle as pkl
from pathlib import Path
from sklearn.model_selection import KFold
import json
from tqdm import tqdm
import copy
import multiprocessing as mp
import numpy as np
from collections import OrderedDict
import os
import lmdb
import shutil
import re
import gc
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # tokenizer 被 fork 的话会报 warning. todo: 探究下如何实现的
device = 'cuda' if T.cuda.is_available() else 'cpu'


def get_embedding(tokenizer, model, txt) -> T.Tensor:
    inputs = tokenizer(txt, return_tensors='pt', padding=True, truncation=True).to(device)
    return model(inputs.input_ids, inputs.attention_mask)


def aug_sample(x):
    a = copy.deepcopy(x)
    a['gt_title'] = a['title']
    a['gt_key_attr'] = a['key_attr'].copy()
    a['gt_match'] = a['match'].copy()
    return trans(a)


if __name__ == "__main__":
    PREPROCESS_OUTPUT.mkdir(exist_ok=True, parents=True)
    prop = get_eq_attr()    # todo: 完善
    prop2id = OrderedDict()  # 由于不同属性下的取值可能相同，所以使用(属性名，属性取值)作为key。等价属性已经替换了。
    prop2uid = OrderedDict() # (属性名，属性取值) -> 唯一 id。等价属性已经替换了。
    for k, v in prop.items():
        for i, v in enumerate(v):
            prop2id[(k, v)] = i
    prop_ncls = [len(v) for _, v in prop.items()]

    tokenizer = AutoTokenizer.from_pretrained("youzanai/bert-product-title-chinese", local_files_only=True)

    all_prop = get_all_keyattr()
    # todo: all_prop.values() 这里有重复的！！
    literals = [tokenizer.special_tokens_map['pad_token'], tokenizer.special_tokens_map['cls_token']] + list(all_prop.keys()) + list(chain.from_iterable(all_prop.values()))  # 用 pad 表示没在标题里匹配到相应属性
    literal2id = OrderedDict({k: i for i, k in enumerate(literals)})

    bert = YZBertTextEnc().to(device)
    literal_embedding_bert = T.cat([get_embedding(tokenizer, bert, k) for k in literals], 0).cpu()
    bert = YZCLIPTextEnc().to(device)
    literal_embedding_clip = T.cat([get_embedding(tokenizer, bert, k) for k in literals], 0).cpu()
    del bert, tokenizer
    gc.collect()

    # make n-fold data
    objs = []
    with open(TRAIN_DATA / 'train_fine.txt', 'r') as fin:
        for x in tqdm(fin.readlines(), desc='read fine'):
            objs.append(std_obj(json.loads(x)))
    with open(TRAIN_DATA / 'train_coarse.txt', 'r') as fin:
        matched = []
        unmatched = []
        for x in tqdm(fin.readlines(), desc='read coarse'):
            x = json.loads(x)
            if x['match']['图文']:
                matched.append(std_obj(x))
            else:
                unmatched.append(x)
    objs += matched
    candidate_title = [x['title'] for x in objs]
    candidate_attr = [x['key_attr'] for x in objs]
    candidate_meta = [x['meta'] for x in objs]
    # todo: candidate_attr、candidate_title 是否应该全保存下来？对于训练集，能否用训练集之外的标题来增广？验证集呢？
    with open(PREPROCESS_OUTPUT / 'props.pkl', 'wb') as fout:
        pkl.dump(dict(
            prop=prop,
            all_prop=all_prop,
            prop2id=prop2id,
            prop_ncls=prop_ncls,
            literals=literals,
            literal_embedding_clip=literal_embedding_clip,
            literal_embedding_bert=literal_embedding_bert,
            literal2id=literal2id,
            candidate_title=candidate_title,
            candidate_attr=candidate_attr,
            candidate_meta=candidate_meta
        ), fout)

    print("Writing Feature DB...")
    fdb = PREPROCESS_OUTPUT / 'feature_db'
    shutil.rmtree(fdb, ignore_errors=True)
    env = lmdb.open(str(fdb), int(1e12), max_dbs=128)
    db = env.open_db(b'train')
    img_names = set()
    with env.begin(db=db, write=True) as txn:
        for i, x in enumerate(tqdm(objs + unmatched, desc='img')):
            if x['img_name'] in img_names:
                if not np.allclose(x['feature'], pkl.loads(txn.get(x['img_name'].encode()))):
                    print('warning: duplicate img name but not same feature! ', x['img_name'])
            else:
                txn.put(x['img_name'].encode(), pkl.dumps(x['feature']))
                img_names.add(x['img_name'])
    print('img done.')
    db2 = env.open_db(b'gttitle')
    with env.begin(db=db2, write=True) as txn:
        for i, x in enumerate(tqdm(objs, desc='title')):
            txn.put(x['img_name'].encode(), pkl.dumps(x['title']))
    db3 = env.open_db(b'gtattr')
    with env.begin(db=db3, write=True) as txn:
        for i, x in enumerate(tqdm(objs, desc='attr')):
            txn.put(x['img_name'].encode(), pkl.dumps(x['key_attr']))
    env.close()

    kf = KFold(5)
    print('Make validation set...')
    trans = sequential_transform([
            random_replace(candidate_attr, candidate_title),
            random_delete(0.3),
            random_modify(0.3, prop)
        ], [0.4, 0.4, 0.4])

    for fd, (train_idx, val_idx) in enumerate(kf.split(range(len(objs)))):
        out_file = PREPROCESS_OUTPUT / f'fold-{fd}.json'
        shutil.rmtree(out_file, ignore_errors=True)
        env = lmdb.open(str(out_file), int(1e12), max_dbs=128)

        db = env.open_db(b'val')
        with env.begin(db=db, write=True) as txn:
            # 线上内存不够，也采用在线增广。
            for i, x in enumerate(val_idx):
                x = objs[x]
                txn.put(str(i).encode(), pkl.dumps({
                    'img_name': x['img_name'],
                    'title': x['title'],
                    'match': x['match'],
                    'key_attr': x['key_attr'],
                    'gt_key_attr': x['key_attr'].copy(),
                    'gt_title': x['title'],
                    'meta': get_meta(x)
                }))
        
        # 训练集暂时采用线上增广的方式
        db = env.open_db(b'train')
        with env.begin(db=db, write=True) as txn:
            for i, x in enumerate(tqdm(train_idx)):
                x = objs[x]
                txn.put(str(i).encode(), pkl.dumps({
                    'img_name': x['img_name'],
                    'title': x['title'],
                    'match': x['match'],
                    'key_attr': x['key_attr'],
                    'gt_key_attr': x['key_attr'].copy(),
                    'gt_title': x['title'],
                    'meta': get_meta(x)
                }))

    print('writing unmatched data...')
    for fd, (train_idx, val_idx) in enumerate(kf.split(range(len(unmatched)))):
        out_file = PREPROCESS_OUTPUT / f'fold-{fd}.json'
        env = lmdb.open(str(out_file), int(1e12), max_dbs=128)

        db = env.open_db(b'unmatched_train')
        with env.begin(db=db, write=True) as txn:
            for i, x in enumerate(tqdm(train_idx)):
                x = unmatched[x]
                txn.put(str(i).encode(), pkl.dumps({
                    'img_name': x['img_name'],
                    'title': x['title']
                }))

        db = env.open_db(b'unmatched_val')
        with env.begin(db=db, write=True) as txn:
            for i, x in enumerate(tqdm(val_idx)):
                x = unmatched[x]
                txn.put(str(i).encode(), pkl.dumps({
                    'img_name': x['img_name'],
                    'title': x['title']
                }))

    print('writing full matched data (not split)...')
    out_file = PREPROCESS_OUTPUT / f'full.json'
    env = lmdb.open(str(out_file), int(1e12), max_dbs=128)
    db = env.open_db(b'train')
    with env.begin(db=db, write=True) as txn:
        for i, x in enumerate(tqdm(objs)):
            txn.put(str(i).encode(), pkl.dumps({
                'img_name': x['img_name'],
                'title': x['title'],
                'match': x['match'],
                'key_attr': x['key_attr'],
                'gt_key_attr': x['key_attr'].copy(),
                'gt_title': x['title'],
                'meta': get_meta(x)
            }))
