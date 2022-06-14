# 所有文件从这里导入需要的文件路径、变量等
import os
import sys
from pathlib import Path

ONLINE = 0
if ONLINE:
    WORK_DATA = Path('/home/mw/work')
    INPUT_DATA = Path('/home/mw/input')
    PROJECT = Path('/home/mw/project')
    PREPROCESS_OUTPUT = Path('/home/mw/temp')
    # 挂载的预处理输出。复现时指定为temp文件夹。注意数据集命名为【jdpreprocessing】，压缩为pre.zip
    PREPROCESS_MOUNT = list(Path('/home/mw/input/').glob('jdpreprocessing*'))[0] / 'pre'
    #PREPROCESS_MOUNT = PREPROCESS_OUTPUT
    
    TEST_DATA = INPUT_DATA / 'track1_contest_4362/'
    TRAIN_DATA = TEST_DATA / 'train/train'
    WEIGHT_OUTPUT = PROJECT / 'best_model'
    SUBMISSION_OUTPUT = PROJECT / 'submission'
    sys.path.append(str((INPUT_DATA / 'trex2485/trexpark').resolve()))
    sys.path.append(str((PROJECT / 'code').resolve()))
    os.environ['TRANSFORMERS_CACHE'] = str((INPUT_DATA / 'hfcache4484/cache').resolve())
else:
    r = Path('/home/renzhen/.jupyter/kaggle/jd2022-semi/')
    WORK_DATA = r / 'work'
    INPUT_DATA = r / 'input'
    PROJECT = r / 'project'
    PREPROCESS_MOUNT = PREPROCESS_OUTPUT = TRAIN_DATA = TEST_DATA = INPUT_DATA / 'data'
    WEIGHT_OUTPUT = PROJECT / 'best_model'
    SUBMISSION_OUTPUT = PROJECT / 'submission'
    sys.path.append(str((r / 'input/trexpark').resolve()))
    sys.path.append(str((PROJECT / 'code').resolve()))
# from prj_config import Path, WORK_DATA, INPUT_DATA, PROJECT, PREPROCESS_OUTPUT, PREPROCESS_MOUNT, TEST_DATA, TRAIN_DATA, WEIGHT_OUTPUT, SUBMISSION_OUTPUT