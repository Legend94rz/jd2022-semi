from prj_config import Path, WORK_DATA, INPUT_DATA, PROJECT, PREPROCESS_OUTPUT, PREPROCESS_MOUNT, TEST_DATA, TRAIN_DATA, WEIGHT_OUTPUT, SUBMISSION_OUTPUT

from fasttorch import T, nn, F, Learner, SparseCategoricalAccuracy, EarlyStopping, ModelCheckpoint, F1Score, LambdaLayer, TensorBoard, BinaryAccuracyWithLogits, GradClipper, StochasticWeightAveraging
from fasttorch.misc.misc import seed_everything, Stage
from fasttorch.metrics.metrics import BaseMeter
from functools import partial
import json
from pathlib import Path
import pickle as pkl
import multiprocessing as mp
import numpy as np
from sklearn.model_selection import KFold
from scipy.special import softmax
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from defs import LmdbObj, MTLearner, ConcatFusion, OnlineMeter, ClassificationPrecision, random_delete, random_modify, random_replace, mutex_transform, sequential_transform, delete_words, replace_hidden, shuffle_title
from tools import is_too_close_negtive, ufset


class JsonDataset(Dataset):
    tokenizer = AutoTokenizer.from_pretrained("youzanai/bert-product-title-chinese", local_files_only=True)

    def __init__(self, literal2id, prop2id, objs, feature_db, prop, trans, enlarge=1):
        # 这里应该只读图文匹配数据，用 trans 变换为不匹配的。
        # 后续的未标记数据也是要预测其正确的标记才能用。
        self.literal2id = literal2id
        self.prop2id = prop2id
        self.prop = prop
        self.objs = objs
        self.feature_db = feature_db
        # trans: 指定如何变换一个 obj
        self.trans = trans
        self.enlarge = enlarge
        
    def __getitem__(self, i):
        a = self.objs[i % len(self.objs)]
        if self.trans:
            while True:
                a = self.trans(a)
                if not is_too_close_negtive(a):
                    break
        # return: (img, q, v, [tt_ids + tt_atmask], qmask, qmatch, gtmask, gtoverall, target)
        # global: literal2id, prop2id, prop
        attr = a['key_attr']
        prop = self.prop
        assert set(a['match']) ^ set(attr) == {'图文'}, f"maybe a wrong data: {a}!!"   # 确保match 项与 attr 只差图文项
        v = [self.literal2id[attr[k]] if k in attr else 0 for k in prop] # 0 是 pad_token，表示没在标题里找到。Nq x 768
        qmask = [int(k in a['key_attr']) for k in prop]  # [Nq]
        qmatch = [a['match'].get(k, 0) for k in prop] # [Nq]
        gtmask = [int(k in a['gt_key_attr']) for k in prop]  # [Nq]
        target = [self.prop2id[(k, ufset[a['gt_key_attr'][k]])] if k in a['gt_key_attr'] else 0 for k in prop]  # 不存在的关键属性可以随机填个数，这里默认是0。 [Nq]
        return self.feature_db[a['img_name']], v, a['title'], qmask, qmatch, gtmask, a['match']['图文'], *target
        
    def __len__(self):
        return int(len(self.objs) * self.enlarge)
    
    @classmethod
    def collate_fn(cls, data):
        feature = T.tensor([x[0] for x in data])
        v = T.tensor([x[1] for x in data])
        inputs = cls.tokenizer([x[2] for x in data], return_tensors='pt', padding=True, truncation=True)  # max_length ??
        qmask = T.tensor([x[3] for x in data], dtype=T.float32)
        qmatch = T.tensor([x[4] for x in data], dtype=T.float32)
        gtmask = T.tensor([x[5] for x in data], dtype=T.float32)
        gtoverall = T.tensor([x[6] for x in data], dtype=T.float32).reshape(-1, 1)
        target = [T.tensor([x[7+i] for x in data]) for i in range(qmask.shape[1])]
        return feature, v, inputs.input_ids, inputs.attention_mask, qmask, qmatch, gtmask, gtoverall, *target
        

def train_model(fd):
    trans = mutex_transform([
        random_replace(candidate_attr, candidate_title),
        random_modify(0.3, prop),
        delete_words(), # 分词后随机选择至少一个短语，删除。相应修改 match 的字段
        replace_hidden(rep_tp=True),  # 随机换类型、颜色或性别中的至少一个，没有这些则保持原输入。
        shuffle_title()
    ], [0.2, 0.2, 0.2, 0.2, 0.2])
    EPOCHS = 26
    feature_db = LmdbObj(PREPROCESS_MOUNT / 'feature_db', 'train')
    #train_obj = LmdbObj(PREPROCESS_MOUNT / f'fold-{fd}.json', 'train')
    train_obj = LmdbObj(PREPROCESS_MOUNT / f'full.json', 'train')
    #val_obj = LmdbObj(PREPROCESS_MOUNT / f'fold-{fd}.json', 'val')
    m = MTLearner(ConcatFusion(prop_ncls, literal_embedding_clip, 'clip'), partial(T.optim.AdamW, lr=4e-4, eps=1e-8), [lambda x: x], amp=True)
    # todo: DDP <--> MP Kfold时 注意修改device
    train_log, val_log = m.fit(JsonDataset(literal2id, prop2id, train_obj, feature_db, prop, trans, 10), EPOCHS, 256, 
        [(None, 'online', OnlineMeter(len(prop_ncls))), (None, 'overall_f1', F1Score(0)), (None, 'clsacc', ClassificationPrecision(len(prop_ncls)))],
        #validation_set=JsonDataset(literal2id, prop2id, val_obj, feature_db, prop, None, 1), 
        callbacks=[StochasticWeightAveraging(1e-6, str(WEIGHT_OUTPUT / f'SimpleFusion-swa-full.pt'), 16, anneal_epochs=10), 
                   #ModelCheckpoint(str(WEIGHT_OUTPUT / f'SimpleFusion-{fd}.pt'), monitor='val_online', mode='max'), 
                   #EarlyStopping('val_online', patience=4, mode='max'), 
                   GradClipper(10)], device=f'cuda:{fd}',
        collate_fn=JsonDataset.collate_fn, num_workers=6, verbose=local_rank == 0, persistent_workers=False, pin_memory=True, prefetch_factor=4, shuffle=False)
    #print(val_log['val_online'].max())


if __name__ == "__main__":
    local_rank = Learner.init_distributed_training(seed=43, dummy=True)
    # 对于 ConcatFusion 模型，amp 好像会导致梯度消失或爆炸，最终输出 nan。 ref: https://github.com/pytorch/pytorch/issues/40497
    # 使用AdamW的一些技巧：https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/288996
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

    train_model(0)
    #procs = []
    #for i in range(5):
    #    p = mp.Process(target=train_model, args=(i, ))
    #    p.start()
    #    procs.append(p)
    #for p in procs:
    #    p.join()
    #    p.close()
