# Netflix 电影时长分析：数据清洗阶段

目标：使用 Elastic Net，根据上映年份、制作国家、影视类型及年龄分级预测电影时长（分钟）。数据不含观众评分，rating 表示年龄分级。

数据来源：https://www.kaggle.com/datasets/paramvir705/netflix-dataset

## 运行

安装 Python 和 pandas 后，在本文件夹中运行：

```bash
python -m pip install pandas
python 01_clean_data.py
```

原始文件 netflix_titles.csv 不会被覆盖。输出 movies_clean.csv 和 cleaning_summary.json。脚本带中文注释，可重复运行。

## 清洗规则

只保留 Movie；对于 duration 缺失、rating 明确为“数字 min”的三条记录，将时长移回 duration，并将未知分级标为 Unknown。时长转换成 duration_min，保留全部有效正数时长，包括短片。国家、分级、类型的缺失标为 Unknown。导演及演员不作为当前模型的特征，缺失不会导致删行。

## 下一步建模

固定随机种子划分训练集和测试集。使用 Pipeline，在训练折内学习国家和类型的多标签编码、年龄分级编码及标准化，再通过交叉验证选择 Elastic Net 参数。不得把 duration 或 duration_min 放入输入特征。与训练集平均时长基线比较测试集 MAE、RMSE、R²，并分析非零系数。当前文件仅完成清洗，尚未训练模型。

## 提交

将 netflix_elastic_net 文件夹放入上周作业所在仓库，与上周作业分文件夹保存。


## 已完成模型训练

第二阶段已完成，请查看 training_results.md；运行 02_train_model.py 可复现训练、预测、指标及图表。
