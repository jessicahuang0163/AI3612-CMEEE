# AI3612  CMeEE: Part 2


## 代码结构

在 `/src` 文件夹下，我们添加的几个 `.py` 文件作用分别为：

|文件|说明|
|---|---|
|`adv_train.py`|实现了对抗训练 FGM/PGD|
|`dataset_analyze.py`|统计了 train, dev, test 集的 text 和 entity 信息并进行可视化|
|`ensemble.py`| 重载了 `Trainer` 以类进行基于参数平均的模型融合|
|`model.py`| 增加了 BERT-BiLSTM-CRF 模型 |
|`plot_loss.py`|利用 sbatch 输出的的 log 文件可视化训练过程中的 loss 和 Micro-F1 变化|
|`utils.py`| 实现了差分学习率和学习率逐层下降 |
|`w2ner.py`|实现了BERT-W2NER Decoder 的整体模型结构|
|`w2ner_dataloader.py`|定义了基于 W2NER 模型的 dataset 和 dataloader |
|`w2ner_trainer.py`|重载了基于 W2NER 的 Trainer 类|



## 结果复现

如报告中最后一节所述，我们提供了两种最优模型及其复现脚本，分别为基于 BERT-CRF 的 `run_cmeee.sbatch` 和 基于 BERT-W2NER 的 `run_w2ner.sbatch`。

需要在 A100 上复现报告中所述的最优结果，因此首先需要运行

```bash
module load miniconda3/4.10.3
module load cuda/11.3.1
source activate medical
```

之后运行 `sbatch run_cmeee.sbatch` 或 `sbatch run_w2ner.sbatch`。


## 最优结果

分别运行两种最优模型的脚本应该分别复现出如下测试集结果。

|运行脚本|Micro-F1|
|---|---|
|`run_cmeee.sbatch`||
|`run_w2ner.sbatch`||