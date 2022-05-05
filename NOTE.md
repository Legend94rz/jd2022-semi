# todo


20. 全数据+单模SWA；bad case；换正常的bert-base试试；去掉moex；modify增广时适当换顺序；加强replace时判断同类的逻辑。
总参数量5亿，
12层vbert 104,246,029，Total Size (MB): 400 ~ 432.04，
6层vbert 61,718,797, Total Size (MB): 236 ~ 254.63
ConcatFusion 106,263,097, Total Size (MB): 440.01 ~ 742
TitleImg 103,255,809  Total Size (MB): 400 ~ 427.72


# submits

0429

visualbert multilabel, 修复了 random_replace 的一些 bug、数据集去重，重新生成数据。线下5折平均0.949091428。14~24个 epoch 达到最优
3 层线下0.9477315480000001；6层线下0.9489622520000001

6层yzvbert lr2e-5, 无reducelr，moex+dropout0.1, 5折线下0.9477151420000001（1）
6层yzvbert lr2e-5, reducelr，无moex，无dropout, 5折线下0.9464627959999999（2）
6层vbert lr2e-5, reducelr，无moex，无dropout, 似乎比（2）稍低些（也可能几乎一样）。没跑完
6层vbert lr2e-5, reducelr，无moex，dropout 0.05。5折线下0.94825013
6层vbert lr2e-5, reducelr，moex+dropout0.1+shuffle_title, 5折线下0.949331714

0430 - 2
visualbert multilabel, 用全量数据+swa，没有验证集，26个epoch，线上0.91462134。
这个版本后来因为线上preprocessing并行，线上oom，改成了在线增广。

0502 - 1
用旧的本地训练权重 pairwise-no-extra-neg-swa，5折，线上0.913

0503-1
pairwise-no-extra-neg. `python -m pairwise.train`, 调整swa参数，排除过于接近的替换标题，全量数据。线上0.9083197004136927。线下训练集0.97083.

0503-2
stage1 model 全量数据+swa. 0.9094251943446834. 训练26个epoch，metric_online=0.97273. 这个版本修复了delete_words和random_replace的一些问题，
multilabel的验证集需要重新生成、模型需要重新训练。

重新生成数据后，5折线下0.9512899499999999

0504-1
用全量新数据重新训练了vbert，线上0.9133395963085875；

0504-2
融合{0503-1}{0503-2}{0504-1} => 0.923

0505-1
vbert multilabel, 改成replace_hidden(rep_color=False, rep_tp=True)，不再进行颜色增广。全量数据，26个epoch, 16+10swa。线下训练集0.98378。线上0.9136121902986771。
似乎确实不能增广颜色。

0505-2
vbert multilabel, 回退了random_replace的修改。只有当有属性交集时去掉替换标题的属性。重新预处理，全量数据，26个epoch, 16+10swa。线下训练集0.98199。线上0.9139804496067645。
