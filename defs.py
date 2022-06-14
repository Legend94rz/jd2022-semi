from prj_config import Path, WORK_DATA, INPUT_DATA, PROJECT, PREPROCESS_OUTPUT, TEST_DATA, TRAIN_DATA, WEIGHT_OUTPUT, SUBMISSION_OUTPUT

import torch as T
import json
from collections import defaultdict
from itertools import chain
from functools import partial
from typing import Dict, List, Union
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import fasttorch as ft
from fasttorch import T, nn, F, Learner, SparseCategoricalAccuracy, EarlyStopping, ModelCheckpoint, F1Score, LambdaLayer
from fasttorch.misc.misc import seed_everything, Stage
from fasttorch.metrics.metrics import BaseMeter
from torch import distributed
from tqdm import tqdm
import datetime as dt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoConfig, VisualBertModel
from tools import get_all_keyattr, read_label_data, extract_prop_from_title, is_equal, ufset, extract_color, extract_type, types, colors, attr_is_intersection
import random
import copy
import matplotlib.pyplot as plt
from src.clip import clip
import warnings
import lmdb
import pickle as pkl
from pathlib import Path
import jieba
import re
seed_everything(43)
for w in sum(list(get_all_keyattr().values()),[]):
    jieba.add_word(w)
for y in range(2015, 2023):
    jieba.add_word(f'{y}年')
for k, v in types.items():
    for w in v:
        jieba.add_word(w)
for k, v in colors.items():
    for w in v:
        jieba.add_word(w)
for w in ['双排扣', '单排扣', '羊绒衫', '加绒裤']:
    jieba.add_word(w)


class YZCLIPTextEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = clip.ClipChineseModel.from_pretrained('youzanai/clip-product-title-chinese', local_files_only=True).eval().requires_grad_(False)
        #self.enc.text_model.pooler.train().requires_grad_(True)
        #self.enc.text_projection.train().requires_grad_(True)
    
    def forward(self, input_ids, attention_mask):
        return self.enc.get_text_features(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False)


class YZBertTextEnc(nn.Module):
    def __init__(self, trainable=False):
        super().__init__()
        self.enc = AutoModel.from_pretrained("youzanai/bert-product-title-chinese", local_files_only=True)
        if not trainable:
            self.enc = self.enc.eval().requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        return self.enc(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False).pooler_output


def get_vbert(pretrained_bert='youzanai/bert-product-title-chinese', nlayer=None):
    bert = AutoModel.from_pretrained(pretrained_bert, local_files_only=True)
    cfg = AutoConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre", local_files_only=True)
    cfg.vocab_size = bert.config.vocab_size
    vbert = VisualBertModel(cfg)
    vbert.embeddings.load_state_dict(bert.embeddings.state_dict(), strict=False)
    vbert.encoder.load_state_dict(bert.encoder.state_dict())
    vbert.pooler.load_state_dict(bert.pooler.state_dict())
    if nlayer is not None:
        vbert.encoder.layer = vbert.encoder.layer[:nlayer]
    return vbert


class VisualBert(nn.Module):
    def __init__(self, ncls=1, nlayer=None, pretrained_bert='youzanai/bert-product-title-chinese'):
        super().__init__()
        self.vbert = get_vbert(pretrained_bert, nlayer=nlayer)
        self.img_prj = nn.Sequential(MoEx1d(0.1), nn.Dropout(0.1))
        #self.img_prj = nn.Dropout(0.05)
        #self.img_prj = MoEx1d(0.3)
        self.cls = nn.Linear(768, ncls)

    def forward(self, img, tt_id, tt_atmask):
        img = self.img_prj(img).unsqueeze(1)
        #img = img.unsqueeze(1)
        o = self.vbert(
            input_ids=tt_id, 
            attention_mask=tt_atmask, 
            visual_embeds=img, 
            visual_token_type_ids=T.ones(img.shape[:-1], dtype=T.long, device=img.device),
            visual_attention_mask=T.ones(img.shape[:-1], dtype=T.float32, device=img.device)
        )
        return self.cls(o.pooler_output)


class MoEx1d(nn.Module):
    # only support LayerNorm(1d)
    def __init__(self, p):
        """
        Args:
            p (_type_): 增广概率
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            with T.no_grad():
                std = x.std(-1, unbiased=False, keepdim=True)
                mean = x.mean(-1, keepdim=True)
                x = (x - mean) / std
                mask = T.distributions.binomial.Binomial(1, probs=T.tensor([self.p])).sample((len(x), ))  # 决定增广哪些样本
                indices = T.stack([T.arange(0, len(x), 1), T.tensor(np.random.choice(len(x), len(x)))]).T
                select = indices.gather(1, mask.long()).reshape(-1)
                x = x * std[select] + mean[select]
        return x

    
class OnlineMeter(BaseMeter):
    def __init__(self, Nq):
        self.Nq = Nq
        self.reset()
        
    def reset(self):
        self.correct = T.zeros(self.Nq+1)
        self.total = T.zeros(self.Nq+1) + 1e-7
        self.one = T.zeros(1)
        return self

    def __call__(self, match, overall, qmask, qmatch, gtoverall):
        inputs = (T.cat([match, overall], -1) > .0).float()  #　注意分类阈值
        targets = T.cat([qmatch, gtoverall], -1)
        mask = T.cat([qmask, T.ones_like(gtoverall)], -1)
        self.correct += ((inputs==targets).float() * mask).sum(0)
        self.total += mask.sum(0)
        self.one += gtoverall.sum()
        return self.value
    
    def sync(self):
        distributed.all_reduce(self.correct, op=distributed.ReduceOp.SUM)
        distributed.all_reduce(self.total, op=distributed.ReduceOp.SUM)

    @property
    def value(self):
        a, b = self.correct[:-1].sum()/self.total[:-1].sum() * 0.5, self.correct[-1] / self.total[-1] * 0.5
        #return T.tensor([a+b, a, b, self.one/self.total[-1]])
        return a+b


# 所有变换方式独立应用于每个属性. p1控制整体是否应用。p2控制每个属性的概率。
# 所有的变换应保证match的key只比key_attr的key多图文项。match用于dataset里计算match，key_attr用于计算query mask 和 query value.
# key_attr:{k: v} 与 match{k: 0/1} 共同表示这个 query k, 如果取值为 v 的话，标题与图片是否匹配，训练数据不一定会保证所有的 v 都出现在标题里、不保证所有的 k 在标题中都可以找到对应项（因为会加混淆），也不保证图片一定有 k 这个关键属性。
#   但在测试过程，v 的取值需要从标题里提取。这里存在一个问题：测试数据能否保证 query 对应的某个 value 一定在标题里？
# 变换方式一：随机删除匹配属性，不影响图文匹配结果。这种方式可能作用不大
# 删除操作最重要的是对于那些既有匹配属性又有不匹配属性的样本，不能把匹配的或不匹配的全删掉。
class random_delete:
    def __init__(self, p2):
        self.p2 = p2

    def __call__(self, obj):
        matched_k = [k for k in obj['key_attr'] if obj['match'][k]]
        unmatched_k = [k for k in obj['key_attr'] if not obj['match'][k]]
        ms = [k for k in matched_k if np.random.random() < self.p2][:len(matched_k) - 1]
        ums = [k for k in unmatched_k if np.random.random() < self.p2][:len(unmatched_k) - 1]
        for k in ms + ums:
            v = obj['key_attr'][k]
            obj['title'] = obj['title'].replace(v, '')  # 从标题中删除
            obj['key_attr'].pop(k)  # 从关键属性中删除
            obj['match'].pop(k)  # 从匹配中删除
        return obj
    
    
# 方式二： 随机修改匹配属性，图文一定会改成不匹配。
# Q: 有无必要修改不匹配属性？
# A1: 若改的话需要备份 ground truth，这样可以避免改成匹配。
# A2: 目前 obj 都是匹配数据，不存在修改不匹配属性的情况。
# 不要把不匹配属性全改成匹配。匹配属性可以全改。
class random_modify:
    def __init__(self, p2, prop):
        self.p2 = p2
        self.prop = prop
        
    def __call__(self, obj):
        matched_k = [k for k in obj['key_attr'] if obj['match'][k]]
        unmatched_k = [k for k in obj['key_attr'] if not obj['match'][k]]
        ms = [k for k in matched_k if np.random.random() < self.p2]
        ums = [k for k in unmatched_k if np.random.random() < self.p2][:len(unmatched_k) - 1]
        for k in ms + ums:
            v = obj['key_attr'][k]
            mv = random.choice(list(set(self.prop[k]) - {v}))
            obj['title'] = obj['title'].replace(v, mv)
            obj['key_attr'][k] = mv
            f = int( is_equal(mv, obj['gt_key_attr'].get(k)) )
            obj['match'][k] = f
            if not f:
                obj['match']['图文'] = 0
        return obj


# 方式三：随机替换标题。图文几乎一定是不匹配。
class random_replace:
    def __init__(self, candidate_key_attr: List[Dict[str, str]], candidate_title: List[str], prop=None, grouped_df: pd.DataFrame = None):
        self.candidate_key_attr = candidate_key_attr
        self.candidate_title = candidate_title
        self.prop = prop
        self.grouped_df = grouped_df
        
    def __call__(self, obj):
        # 第一种替换。随机找一个。
        i = np.random.choice(len(self.candidate_title))
        newtitle = self.candidate_title[i]
        attr = self.candidate_key_attr[i].copy()

        if self.grouped_df is not None:
            assert self.prop is not None
            color = extract_color(obj['gt_title'])
            tp = extract_type(obj['gt_title'])
            query = ' and '.join([f'{k}=="{v}"' for k, v in obj['gt_key_attr'].items()])
            if np.random.random() < 0.6 and (color is not None or tp is not None) and len(query) > 0:
                # 第二种替换。保持属性相同，找一个：颜色或类型不同的
                candidate = []
                ne = []
                if color:
                    ne.append(f"(颜色!='{color}' and 颜色.notna())")
                if tp:
                    ne.append(f"(类型!='{tp}' and 类型.notna())")
                ne = '(' + ' or '.join(ne) + ')'
                tt = obj["title"]
                tmp = self.grouped_df.query(f'{query} and {ne} and title!="{tt}"', engine='python')
                if len(tmp) > 0:
                    prob = tmp['title'].map(len).values
                    prob = prob / prob.sum()
                    i = np.random.choice(len(tmp), p=prob, replace=False)
                    row = tmp.iloc[i]
                    attr = {k: row[k] for k in self.prop if not pd.isna(row[k])}
                    newtitle = np.random.choice(row['title'])

        # key_attr的 values 都在新替换的标题里
        obj['match'] = {k: int( is_equal(obj['gt_key_attr'].get(k), attr[k]) ) for k in attr}
        obj['match']['图文'] = 0
        # 如果替换前后属性（从描述上）属于同一大类（上衣、裤子、包、鞋）。这时候应该去掉原标题中未出现的属性，防止过多的不匹配
        if attr_is_intersection(attr, obj['gt_key_attr']):
            keys = list(attr.keys())
            for k in keys:
                if k not in obj['gt_key_attr']:
                    newtitle = newtitle.replace(attr[k], '')
                    attr.pop(k)
                    obj['match'].pop(k)

        obj['title'] = newtitle
        obj['key_attr'] = attr
        return obj


# 根据图片的gt, 添加一些标题中没出现的属性，不影响图文匹配。
class random_add_words:
    def __init__(self, p):
        self.p = p

    def __call__(self, obj):
        self_words = jieba.lcut(obj['title'])
        ori_words = jieba.lcut(obj['gt_title'])
        gt_attr = obj['gt_key_attr']
        inv_gt_attr = {v: k for k, v in gt_attr.items()}
        no_appear = list(set(ori_words) - set(self_words))
        t = obj['title']
        for w in no_appear:
            if np.random.random() < self.p:
                if w in inv_gt_attr:# 如果要添加的词是一个属性值
                    if inv_gt_attr[w] in obj['key_attr']: # 如果该属性已经在标题里出现过了，不能添加，跳过。
                        continue
                    obj['key_attr'][inv_gt_attr[w]] = w
                    obj['match'][inv_gt_attr[w]] = 1
                if np.random.random() < 0.5:
                    t = w + t
                else:
                    t = t + w
        obj['title'] = t
        return obj


class delete_words:
    def __init__(self):
        self.colors = set.union(*[v for v in colors.values()])
        self.types = set.union(*[v for v in types.values()])

    def __call__(self, obj):
        if not obj['match']['图文']:
            return obj
        words = jieba.lcut(obj['title'])
        pvalue = set(obj['key_attr'].values())
        # 至少保留类型、颜色、性别、属性中的一个。如果一个都没有则不增广。
        # 颜色 与 类型 按照颜色表与类型表来识别；属性按照属性表
        # 在剩下的短语中选择 N 个删除。
        type_id = []
        color_id = []
        gender_id = []
        property_id = []
        other_id = []
        for i, w in enumerate(words):
            if w in self.types:
                type_id.append(i)
            elif w in self.colors:
                color_id.append(i)
            elif w in pvalue:
                property_id.append(i)
            elif w in {'女装', '女士', '男装', '男士'}:
                gender_id.append(i) 
            else:
                other_id.append(i)
        # 确保隐藏属性只有一个，保证是准确的
        info = (type_id if len(type_id) == 1 else []) + (color_id if len(color_id) == 1 else []) + (gender_id if len(re.findall('女|男', obj['title'])) == 1 else []) + property_id
        if len(info) < 1:
            return obj
        keep = sorted(np.random.choice(info, np.random.randint(1, len(info)+1), replace=False).tolist() + np.random.choice(other_id, np.random.randint(0, len(other_id)+1), replace=False).tolist())
        property_id = set(property_id)
        new_title = ''
        new_match = {'图文': 1}
        new_key_attr = {}
        for i in keep:
            new_title += words[i]
        for k, v in obj['key_attr'].items():
            if v in new_title:
                new_match[ k ] = 1
                new_key_attr[ k ] = v
        obj['title'] = new_title
        obj['key_attr'] = new_key_attr
        obj['match'] = new_match
        return obj


class replace_hidden:
    def __init__(self, rep_color=False, rep_tp=False, rep_gender=False):
        self.colors = '|'.join(set.union(*[v for v in colors.values()]))
        self.inv_color = {w: k for k, v in colors.items() for w in v}
        self.types = '|'.join(set.union(*[v for v in types.values()]))
        self.inv_type = {w: k for k, v in types.items() for w in v}
        self.rep_color = rep_color
        self.rep_tp = rep_tp
        self.rep_gender = rep_gender

    def _color_repl(self, title):
        color = re.findall(self.colors, title)
        if len(color) == 1:
            color = color[0]
            # 不同组的换，防止差异太小
            return color, np.random.choice(list(set(colors) - {self.inv_color[color]}))
        return None

    def _type_repl(self, title):
        tp = re.findall(self.types, title)
        if len(tp) == 1:
            tp = tp[0]
            # 同组里的换，防止差距过大
            return tp, np.random.choice(list(types[self.inv_type[tp]]))
        return None

    def _gender_repl(self, title):
        g = re.findall('女|男', title)
        if len(g) == 1:
            return g[0], ({'女', '男'} - {g[0]}).pop()
        return None

    def __call__(self, obj):
        # 随机换类型、颜色或性别中的至少一个，没有这些则保持原输入。图文一定改成不匹配
        # 对于颜色，需要先确定颜色是哪个大类，修改为另一个大类里的。如果无法确定属于哪个大类则不替换。
        #words = jieba.lcut(obj['title'])
        title = obj['title']
        color = self._color_repl(title) if self.rep_color else None
        tp = self._type_repl(title) if self.rep_tp else None
        gender = self._gender_repl(title) if self.rep_gender else None
        choices = [x for x in [color, tp, gender] if x is not None]
        if choices:
            for x in np.random.choice(len(choices), np.random.randint(1, len(choices)+1), replace=False):
                x = choices[x]
                title = title.replace(x[0], x[1])
            obj['title'] = title
            obj['match']['图文'] = 0
        return obj


# 把key_attr的value把title切开，然后打乱顺序。不影响图文
class shuffle_title:
    def __call__(self, obj):
        if len(obj['key_attr']):
            words = re.split('|'.join([f'({v})' for v in obj['key_attr'].values()]), obj['title'])
            words = [w for w in words if w]
            np.random.shuffle(words)
            obj['title'] = ''.join(words)
        return obj


class sequential_transform:
    def __init__(self, trans, p):
        self.trans = trans
        self.p = p
        
    def __call__(self, obj):
        for i, t in enumerate(self.trans):
            if np.random.random() < self.p[i]:
                obj = t(obj)
        return obj
    
    
class mutex_transform:
    def __init__(self, trans, p):
        self.trans = trans
        self.p = p
        
    def __call__(self, obj):
        i = np.random.choice(len(self.trans), p=self.p)
        #print(f'choose {i}')
        return self.trans[i](obj)
