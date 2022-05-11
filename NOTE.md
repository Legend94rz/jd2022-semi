# todo


20. 全数据+单模SWA；bad case；换正常的bert-base试试；去掉moex；modify增广时适当换顺序；加强replace时判断同类的逻辑。
总参数量5亿，
12层vbert 104,246,029，Total Size (MB): 400 ~ 432.04，
6层vbert 61,718,797, Total Size (MB): 236 ~ 254.63
ConcatFusion 106,263,097, Total Size (MB): 440.01 ~ 742
TitleImg 103,255,809  Total Size (MB): 400 ~ 427.72

规则后处理图文1而有属性不匹配的；stacking+pairwise；


# submits

0429

visualbert multilabel, 修复了 random_replace 的一些 bug、数据集去重，重新生成数据。线下5折平均0.949091428。14~24个 epoch 达到最优
3 层线下0.9477315480000001；6层线下0.9489622520000001

6层yzvbert lr2e-5, 无reducelr，moex+dropout0.1, 5折线下0.9477151420000001（1）
6层yzvbert lr2e-5, reducelr，无moex，无dropout, 5折线下0.9464627959999999（2）
6层vbert lr2e-5, reducelr，无moex，无dropout, 似乎比（2）稍低些（也可能几乎一样）。没跑完
6层vbert lr2e-5, reducelr，无moex，dropout 0.05。5折线下0.94825013
6层vbert lr2e-5, reducelr，moex+dropout0.1+shuffle_title, 5折线下0.949331714
<0507>6层vbert lr2e-5, reducelr，moex+dropout0.1 update random_delete(但没有重新生成验证集数据), no shuffle_title, 5折线下0.94989896

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


0506-1
pairwise. 重写了数据生成逻辑，使用在线方式，用6层YZbert初始化。5折线下0.949399712。线上5折0.9101974132836297

0506-2<0507-1>
带swa的线上：0.90922021184707 .

0506这两个提交分别是交的ModelCheckpoint和SWA的权重。0506-2线上提交出错了一次，实际提交在0507-1。

0507-2

重写random_delete，匹配属性可以全删. 重新训练vbert + swa, 5折的数据见最上面。全量数据线下训练集0.98233，线上 0.9117030095012957

0508-1
用重写的random_delete,全量数据重新训练pairwise+swa，10epoch=6+4, 线上0.9026880961040142，线下训练集0.97119。

0509-1
回退random_delete，multilabel vbert 加 kl正则，5折线下0.95006181。重新全量数据训练，线下训练集0.98210，线上0.916362394872028

顺便试了一个fasttext(globalpool, 取前250个词频最高的当词典)，与mlp(img)直接concat，二分类acc~77-78

0509-2
融合{0503-1}{0503-2}{0509-1} => 0.9241852957221439

0510-2
方式理论上与0506-1的是一样的，这次是全量数据+swa，线上0.9017562531254262，可能是epoch太多，过拟合了。

0511-1
concat bert. epoch 18+12, swa. 全量数据，线下训练集0.96516, 线上0.8905333522753103 

0511-2
12层pairwise。去掉了对make_sample的random_add，5+2 swa，全量数据，线下训练集0.96928，线上0.8957516456789563 
