from prj_config import Path, WORK_DATA, INPUT_DATA, PROJECT, PREPROCESS_OUTPUT, PREPROCESS_MOUNT, TEST_DATA, TRAIN_DATA, WEIGHT_OUTPUT, SUBMISSION_OUTPUT

from fasttorch import T, nn, F, Learner, SparseCategoricalAccuracy, EarlyStopping, ModelCheckpoint, F1Score, LambdaLayer
from fasttorch.misc.misc import seed_everything, Stage
from fasttorch.metrics.metrics import BaseMeter
from torch.utils.data import Dataset
from functools import partial
from collections import defaultdict
from transformers import AutoTokenizer
import json
from pathlib import Path
import pickle as pkl
import datetime as dt
from tqdm import tqdm
import copy
import numpy as np
import pandas as pd
from scipy.special import softmax, expit as sigmoid
import argparse
from transformers import AutoTokenizer
import warnings
from defs import MTLearner, ConcatFusion
from tools import extract_prop_from_title, attr_to_oh, compute_score, write_submit, ufset, std_obj


class JsonTestDataset(Dataset):
    tokenizer = AutoTokenizer.from_pretrained("youzanai/bert-product-title-chinese", local_files_only=True)

    def __init__(self, literal2id, objs, prop):
        self.literal2id = literal2id
        self.prop = prop
        self.objs = objs
        
    def __getitem__(self, i):
        a = self.objs[i]
        # return: (img, q, v, [tt_ids + tt_atmask], qmask, qmatch, gtmask, gtoverall, target)
        # global: literal2id, prop2id, prop
        prop = self.prop
        if 'query' in a:
            query = set(a['query'])
            attr = extract_prop_from_title(a['title'])
            if not (query - {'图文'}).issubset(set(attr.keys())):
                warnings.warn(f"Query an inexist property. img:{a['img_name']}, query: {query}, title: {a['title']}, attr: {attr}")
        else:
            # 为方便本地验证
            query = set(a['key_attr'].keys())
            attr = a['key_attr']
        # query mask: query 了哪些属性
        # v: 对于每个 query，标题里的是什么
        # k in attr and k in query 这个条件，用于当 a 有 query 字段时，确保 v 里的每个元素都是被 query ，且出现在标题里的。
        # 验证：如果漏掉 k in query，那可能就会把标题里出现但没 query 的当输出。
        # 如果漏掉 k in attr，那可能attr[k]找不到，报错。
        v = [self.literal2id[attr[k]] if k in attr else 0 for k in prop]  # 0 是 pad_token，表示没在标题里找到。Nq x 768, Tensor
        return a['feature'], v, a['title']
        
    def __len__(self):
        return len(self.objs)
    
    @classmethod
    def collate_fn(cls, data):
        feature = T.tensor([x[0] for x in data])
        v = T.tensor([x[1] for x in data])
        inputs = cls.tokenizer([x[2] for x in data], return_tensors='pt', padding=True, truncation=True)  # max_length ??
        return feature, v, inputs.input_ids, inputs.attention_mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-w', '--weight_files', type=str, nargs='+', required=True)
    parser.add_argument('-o', '--output_file', type=str, required=True)
    return parser.parse_args()


def inference(prop, test_obj):
    # todo: 调阈值。可能需要与metric同步改。
    # todo: 并行
    # 集成多个模型
    matches = []
    overalls = []
    has_props = []
    logitss = [[] for _ in range(len(prop))]
    for i, pt in enumerate(args.weight_files):
        pt = Path(pt)
        if pt.exists():
            m = MTLearner(ConcatFusion(prop_ncls, literal_embedding_clip, 'clip'), amp=True).load(pt, 'cuda')
            match, has_prop, overall, *logits = m.predict(JsonTestDataset(literal2id, test_obj, prop), batch_size=256, device='cuda', collate_fn=JsonTestDataset.collate_fn)
            matches.append(sigmoid(match))
            overalls.append(sigmoid(overall))
            has_props.append(sigmoid(has_prop))
            assert len(prop) == len(logits)
            for j in range(len(logits)):
                logitss[j].append(softmax(logits[j], axis=-1))
        else:
            print(f"WARN: cannot find weights: {pt}")

    match = np.mean(matches, 0)
    has_prop = np.mean(has_props, 0)
    overall = np.mean(overalls, 0)
    for j in range(len(prop)):
        logitss[j] = np.mean(logitss[j], 0)
    
    submit = []
    keys = list(prop)
    for i in range(len(test_obj)):
        a = test_obj[i]
        all_match = {keys[j]: match[i, j] for j in range(len(keys))}
        all_match['图文'] = overall[i].item()
        # todo: 再理清下from_title_oh和assume_oh
        if 'query' in a:
            mkeys = query = a['query']
            attr = extract_prop_from_title(a['title'])
            from_title_oh = attr_to_oh({k: ufset[v] for k, v in attr.items()}, prop, prop2id)
            assume_oh = attr_to_oh({k: ufset[attr[k]] for k in query if k in attr}, prop, prop2id)
        else:
            query = list(a['key_attr'].keys())
            mkeys = list(a['match'].keys())
            attr = extract_prop_from_title(a['title'])
            from_title_oh = attr_to_oh({k: ufset[v] for k, v in attr.items()}, prop, prop2id)
            assume_oh = attr_to_oh({k: ufset[v] for k, v in a['key_attr'].items()}, prop, prop2id)

        submit.append({
            'img_name': a['img_name'],
            'match': {k: float(all_match[k]) for k in mkeys},
            'additional': {
                'query': [int(k in query) for k in keys], # 与 match对应的，query mask。
                'assume_oh': assume_oh.tolist(), # key_attr或 标题提取的与 query 的交集（测试时）。one-hot 形式，长度与 logits 一样。
                'from_title_oh': from_title_oh.tolist(), # 标题里。one-hot 形式，长度与 logits 一样。
                'match': match[i].tolist(), # match 的概率
                'has_prop': has_prop[i].tolist(), # has_prop
                'overall': overall[i].tolist(), # overall 的概率
                'logits': np.concatenate([logitss[j][i] for j in range(len(keys))]).tolist(), # concate后的各属性概率
                'key_attr': {keys[j]: prop[keys[j]][logitss[j][i].argmax()] for j in range(len(keys)) if has_prop[i][j] > 0.5}, # 由图片输出的，有哪些属性，及其取值（用于 bad case 分析）
            }
        })
    return submit


if __name__ == "__main__":
    args = parse_args()
    with open(PREPROCESS_MOUNT / 'props.pkl', 'rb') as fin:
        data = pkl.load(fin)
        prop = data['prop']
        all_prop = data['all_prop']
        prop2id = data['prop2id']
        prop_ncls = data['prop_ncls']
        literals = data['literals']
        literal_embedding_bert = data['literal_embedding_bert']
        literal_embedding_clip = data['literal_embedding_clip']
        literal2id = data['literal2id']
        candidate_title = data['candidate_title']
        candidate_attr = data['candidate_attr']

    online_test_obj = []
    with open(args.input_file, 'r') as fin:
        for x in tqdm(fin.readlines(), desc='read test'):
            online_test_obj.append(std_obj(json.loads(x)))
    submit_objs = inference(prop, online_test_obj)
    write_submit(submit_objs, args.output_file, ignore_additional=True)
