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