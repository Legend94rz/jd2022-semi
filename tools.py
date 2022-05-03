import json
from collections import defaultdict, OrderedDict
from itertools import chain
from typing import Dict, List
import pandas as pd
import numpy as np
import re
from pathlib import Path
ufset = {
    '高领': '高领', '半高领': '高领', '立领': '高领', '连帽': '连帽', '可脱卸帽': '连帽', '翻领': '翻领', '衬衫领': '翻领', 'polo领': '翻领', '方领': '翻领', '娃娃领': '翻领', '荷叶领': '翻领', '双层领': '双层领', '西装领': '西装领', 'u型领': 'u型领', '一字领': '一字领', '围巾领': '围巾领', '堆堆领': '堆堆领', 'v领': 'v领', '棒球领': '棒球领', '圆领': '圆领', '斜领': '斜领', '亨利领': '亨利领', '短袖': '短袖', '五分袖': '短袖', '九分袖': '九分袖', '长袖': '九分袖', '七分袖': '七分袖', '无袖': '无袖', '超短款': '超短款', '短款': '超短款', '常规款': '超短款', '长款': '长款', '超长款': '长款', '中长款': '中长款', '修身型': '修身型', '标准型': '修身型', '宽松型': '宽松型', '短裙': '短裙', '超短裙': '短裙', '中裙': '中裙', '中长裙': '中裙', '长裙': '长裙', '套头': '套头', '开衫': '开衫', '手提包': '手提包', '单肩包': '单肩包', '斜挎包': '斜挎包', '双肩包': '双肩包', 'o型裤': 'o型裤', '锥形裤': 'o型裤', '哈伦裤': 'o型裤', '灯笼裤': 'o型裤', '铅笔裤': '铅笔裤', '直筒裤': '铅笔裤', '小脚裤': '铅笔裤', '工装裤': '工装裤', '紧身裤': '紧身裤', '背带裤': '背带裤', '喇叭裤': '喇叭裤', '微喇裤': '喇叭裤', '阔腿裤': '阔腿裤', '短裤': '短裤', '五分裤': '五分裤', '七分裤': '七分裤', '九分裤': '九分裤', '长裤': '九分裤', '松紧': '松紧', '拉链': '拉链', '系带': '系带', '松紧带': '松紧带', '套筒': '套筒', '套脚': '套筒', '一脚蹬': '套筒', '魔术贴': '魔术贴', '搭扣': '搭扣', '高帮': '高帮', '中帮': '高帮', '低帮': '低帮',
    #'男': '男', '女': '女',
    #'黑色': '黑色', '白色': '白色', '灰色': '灰色', '蓝色': '蓝色', '红色': '红色', '粉色': '粉色', '绿色': '绿色', '黄色': '黄色', '棕色': '棕色'
}
types = {
    '上衣': set('毛呢大衣|风衣|polo衫|打底衫|针织衫|t恤|羽绒服|卫衣|皮衣|马甲|羊毛衫|雪纺衫|皮草|连衣裙|棉服'.split('|')),
    '裤子': set('运动裤|正装裤|牛仔裤'.split('|')),
    '包': set('女包|男包|公文包|运动包'.split('|')),
    '鞋': set('休闲鞋|篮球鞋|板鞋|登山鞋|帆布鞋|皮鞋|马丁靴|工装鞋'.split('|'))
}
colors = {
    '红色': set('红色|大红色|酱红色|橘红色|粉红色|浅粉色|粉色|砖红色|酒红色'.split('|')),
    '橙色': set('橙色|金色|焦糖色|杏色'.split('|')), 
    '黄色': set('咖啡色|卡其色|褐色|鹅黄色'.split('|')), 
    '绿色': set('绿色|墨绿色|橄榄绿'.split('|')),
    '蓝色': set('灰蓝色|蓝色|浅蓝色|海蓝色|粉蓝色|天蓝色|海蓝色'.split('|')), 
    '青色': {'藏青色'},
    '紫色': set('紫色|暗紫色'.split('|')), 
    '黑色': {'黑色'},
    '白色': set('白色|米色|银色|米白色'.split('|')), 
    '灰色': set('灰色|浅灰色'.split('|'))
}
attr_type_map = {
    '裤型': 'A', '裤长': 'A', '裤门襟': 'A',
    '版型': 'B', '穿着方式': 'B', '袖长': 'B', '衣长': 'B', '领型': 'B', '裙长': 'B',
    '闭合方式': 'C', '鞋帮高度': 'C',
    '类别': 'D'
}

def get_all_keyattr() -> Dict[str, List[str]]:
    return OrderedDict({
        '领型': ['polo领', '可脱卸帽', '半高领', '衬衫领', '娃娃领', '荷叶领', '双层领', '西装领', 'u型领', '一字领', '围巾领', '堆堆领', '棒球领', '亨利领', '高领', '立领', '连帽', '翻领', '方领', 'v领', '圆领', '斜领'],
        '袖长': ['五分袖', '九分袖', '七分袖', '短袖', '长袖', '无袖'],
        '衣长': ['超短款', '常规款', '超长款', '中长款', '短款', '长款'],
        '版型': ['修身型', '标准型', '宽松型'],
        '裙长': ['超短裙', '中长裙', '短裙', '中裙', '长裙'],
        '穿着方式': ['套头', '开衫'],
        '类别': ['手提包', '单肩包', '斜挎包', '双肩包'],
        '裤型': ['o型裤', '锥形裤', '哈伦裤', '灯笼裤', '铅笔裤', '直筒裤', '小脚裤', '工装裤', '紧身裤', '背带裤', '喇叭裤', '微喇裤', '阔腿裤'],
        '裤长': ['五分裤', '七分裤', '九分裤', '短裤', '长裤'],
        '裤门襟': ['松紧', '拉链', '系带'],
        '闭合方式': ['松紧带', '一脚蹬', '魔术贴', '拉链', '套筒', '套脚', '系带', '搭扣'],
        '鞋帮高度': ['高帮', '中帮', '低帮'],
        #'性别': ['男', '女'], 
        #'颜色': ['黑色', '白色', '灰色', '蓝色', '红色', '粉色', '绿色', '黄色', '棕色']
    })


def get_eq_attr()-> Dict[str, List[str]]:
    return OrderedDict({
        '领型': ['棒球领', '一字领', '围巾领', '西装领', '堆堆领', 'u型领', '双层领', '亨利领', 'v领', '连帽', '圆领', '翻领', '斜领', '高领'],
        '袖长': ['九分袖', '七分袖', '短袖', '无袖'],
        '衣长': ['中长款', '超短款', '长款'],
        '版型': ['修身型', '宽松型'],
        '裙长': ['中裙', '短裙', '长裙'],
        '穿着方式': ['开衫', '套头'],
        '类别': ['斜挎包', '单肩包', '手提包', '双肩包'],
        '裤型': ['背带裤', '喇叭裤', '铅笔裤', 'o型裤', '阔腿裤', '紧身裤', '工装裤'],
        '裤长': ['七分裤', '五分裤', '九分裤', '短裤'],
        '裤门襟': ['拉链', '系带', '松紧'],
        '闭合方式': ['松紧带', '魔术贴', '系带', '拉链', '套筒', '搭扣'],
        '鞋帮高度': ['高帮', '低帮'],
        #'性别': ['男', '女'],
        #'颜色': ['黑色', '白色', '灰色', '蓝色', '红色', '粉色', '绿色', '黄色', '棕色']
    })


def extract_prop_from_title(t)->Dict[str, str]:
    attr = {}
    prop = get_all_keyattr()
    for k, v in prop.items():
        if k == '裤门襟':
            if '裤' in t:
                for x in v:
                    if x in t:
                        attr[k] = x
                        break
        elif k== '闭合方式':
            if '鞋' in t or '靴' in t:
                for x in v:
                    if x in t:
                        attr[k] = x
                        break
        elif k == '颜色' or k == '性别':
            tmp = re.findall('|'.join(v), t)
            if len(tmp) == 1:
                attr[k] = tmp[0]
        else:
            for x in v:
                if x in t:
                    attr[k] = x
                    break
    return attr


def get_meta(obj):
    return [x for x in re.split('|'.join(obj['key_attr'].values()), obj['title']) if len(x)>0] if obj['key_attr'] else [obj['title']]


def std_obj(obj):
    # 输入一个原始对象{'img_name', 'title', 'feature', 'key_attr', 'match'}
    # 返回一个标准化的。如果对象没有query字段，认为是图文匹配的，返回的带有match项，否则（用于infer），没有match项。
    t = re.sub(r'20[0-9]{2}年', '', obj['title'].lower())
    attr = extract_prop_from_title(t)
    for k, v in attr.items():
        if ufset[v] != v:
            t = t.replace(v, ufset[v])
    ret = {
        'img_name': obj['img_name'], 
        'title': t,
        'feature': obj['feature'],
        'key_attr': {k: ufset[v] for k, v in attr.items()}
    }
    ret['meta'] = get_meta(ret)
    if 'query' in obj:
        ret['query'] = obj['query']
    else:
        ret['query'] = list(attr) + ['图文']
        ret['match'] = {k: 1 for k in ret['query']}
    return ret


def extract_color(title):
    color = set(re.findall('[^纯]色', title))
    if len(color) != 1:
        color = {None}
    return color.pop()


def extract_type(title):
    words = '|'.join(chain.from_iterable(types.values()))
    tp = set(re.findall(words, title))
    if len(tp) != 1:
        tp = {None}
    return tp.pop()


def attr_to_oh(attr, prop, prop2id):
    oh = []
    for p in prop:
        lst = np.zeros(len(prop[p]))
        if p in attr:
            i = prop2id[(p, attr[p])]
            if 0 <= i < len(prop[p]):
                lst[ i ] = 1
        oh.append(lst)
    return np.concatenate(oh)


def is_equal(v1, v2):
    # 返回v1与v2是否是等价属性
    return ufset.get(v1) == ufset.get(v2)


def read_label_data(fine_txt, prop: Dict[str, List]) -> pd.DataFrame:
    # index: img_name, columns: [title, feature(2048), key_attr]]
    info = []
    attr = []
    match = []
    with open(fine_txt, 'r') as fin:
        for x in fin.readlines():
            x = json.loads(x)
            info.append([x['img_name'], x['title'], x['feature']])
            attr.append(x['key_attr'])
            m = x['match']["图文"]
            assert m==1  # 确保都是匹配数据
            assert all(v in x['title'] for v in x['key_attr'].values()), f"{x['img_name']} {x['title']} {x['key_attr']} {x['match']}"  # 确保所有的属性值都出现在标题里
            match.append([m])
            
    df = pd.concat([
        pd.DataFrame(info, columns=['img_name', 'title', 'feature']),
        pd.DataFrame.from_records(attr, columns=list(prop.keys())),
        pd.DataFrame(match, columns=['图文'])
    ], axis=1)
    return df


def read_unlabel_data(coarse_txt, prop:  Dict[str, List[str]]) -> pd.DataFrame:
    # 自动提取标题中相应的属性。由于有属性取值相同，会引入少量噪声。
    # 使用这个做训练的话还要注意去掉图文不匹配的。
    info = []
    attr = []
    match = []
    with open(coarse_txt, 'r') as fin:
        for x in fin.readlines():
            x = json.loads(x)
            info.append([x['img_name'], x['title'], x['feature']])
            m = x['match']["图文"]
            attr.append(extract_prop_from_title(x['title']))
            match.append([m])
            
    df = pd.concat([
        pd.DataFrame(info, columns=['img_name', 'title', 'feature']),
        pd.DataFrame.from_records(attr, columns=list(prop.keys())),
        pd.DataFrame(match, columns=['图文'])
    ], axis=1)
    return df


def read_test_data(test_txt, prop: Dict[str, List[str]]) -> pd.DataFrame:
    info = []
    query = []
    with open(test_txt, 'r') as fin:
        for x in fin.readlines():
            x = json.loads(x)
            info.append([x['img_name'], x['title'], x['feature']])
            query.append({k: -1 for k in x['query']})
            
    df = pd.concat([
        pd.DataFrame(info, columns=['img_name', 'title', 'feature']),
        pd.DataFrame.from_records(query, columns=list(prop.keys()) + ['图文']),
    ], axis=1)
    return df


def compute_score(prop, submit, ground_truth):
    """
    submit: {img_name: 'xx', match: {yyy}, [additional: {zzzz}]}
    ground_truth: {img_name, feature, title, match, key_attr, [gt_title], [gt_match], [gt_key_attr]}
    """
    assert len(submit) == len(ground_truth)
    info = []
    answer = []
    pred_attr = []
    gt_attr = []
    gt_overall = []
    allk = list(prop)
    has_pred_attr = 'additional' in submit[0]

    for i in range(len(submit)):
        pred = submit[i]['match']
        target = ground_truth[i]['match']

        assert set(pred.keys()).issubset(set(target.keys())), f"Unmatched sample pair: {pred}, {target}"
        info.append([ground_truth[i]['img_name'], ground_truth[i]['title'], ground_truth[i]['title'] if target['图文'] else ground_truth[i]['gt_title'] ])
        answer.append({k: pred[k] == target[k] for k in pred})
        gt_overall.append(target['图文'])
        if has_pred_attr:
            pred_attr.append(submit[i]['additional']['key_attr'])
            gt_attr.append(ground_truth[i]['key_attr'] if target['图文'] else ground_truth[i]['gt_key_attr'])
    flag = pd.DataFrame.from_records(answer, columns=allk + ['图文'])
    if has_pred_attr:
        df = pd.concat([
            pd.DataFrame(info, columns=['img_name', 'title', 'gt_title']),
            pd.concat([
                flag.rename(columns={k: k+'f' for k in allk}),
                pd.DataFrame.from_records(pred_attr, columns=allk).rename(columns={k: k+'p' for k in allk}),
                pd.DataFrame.from_records(gt_attr, columns=allk).rename(columns={k: k+'y' for k in allk}),
                pd.DataFrame(gt_overall, columns=['图文y'])
            ], axis=1).sort_index(axis=1)
        ], axis=1)
        bad_case = df[(~pd.isna(flag)).sum(1) != flag.sum(1) ]
    else:
        df = pd.concat([
            pd.DataFrame(info, columns=['img_name', 'title', 'gt_title']),
            pd.concat([
                flag.rename(columns={k: k+'f' for k in allk}),
                pd.DataFrame(gt_overall, columns=['图文y'])
            ], axis=1).sort_index(axis=1)
        ], axis=1)
        bad_case = df[(~pd.isna(flag)).sum(1) != flag.sum(1) ]
    print(flag.sum() / (~pd.isna(flag)).sum()) # 单个属性的acc
    title_score = flag['图文'].sum() / (~pd.isna(flag['图文'])).sum()
    prop_score = flag[allk].sum().sum() / (~pd.isna(flag[allk])).sum().sum()
    score = (title_score + prop_score) / 2
    return score, bad_case


def write_submit(submit_objs, filename, ignore_additional=False):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as fout:
        for x in submit_objs:
            if ignore_additional and 'additional' in x:
                x.pop('additional')
            fout.write(json.dumps(x, ensure_ascii=False) + '\n')
    print(f'writen submits to {filename}!')


def hash_obj(x):
    return f'{x["img_name"]}-{x["title"]}'


def is_too_close_negtive(a):
    # 如果只有图文不匹配，其他所有属性全匹配，但标题是原标题的子集。这说明是一个坏样本
    return not a['match']['图文'] and all(a['match'][k] for k in a['match'] if k!='图文') and set(a['title']).issubset(set(a['gt_title']))


def attr_is_intersection(attra, attrb):
    a = set(attr_type_map[k] for k in attra)
    b = set(attr_type_map[k] for k in attrb)
    return len(a.intersection(b)) > 0
