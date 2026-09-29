# 房价预测（机器学习回归建模）

对区域房屋销售数据完成完整的机器学习建模流程：数据理解、缺失值诊断、相关性分析、地理分布可视化、特征标准化，再对比五种回归模型并用网格搜索调参，最终模型 R² 达到 0.886。

## 数据

- 来源：区域房屋销售数据，共 **21613** 条记录
- 主要字段：`price`（房价）、`bedrooms`、`bathrooms`、`sqft_living`（居住面积）、`sqft_lot`（占地面积）、`floors`、`waterfront`（是否临水）、`view`、`condition`、`grade`、`sqft_above`、`sqft_basement`、`yr_built`、`yr_renovated`、`zipcode`、`lat`、`long`
- 目标变量：`price`

> 数据集未包含在本仓库中。该数据集为公开的 King County（美国金县）房屋销售数据，可在 Kaggle 搜索 "House Sales in King County, USA" 获取。下载后把标准化后的特征表命名为 `test1.csv` 放在项目根目录即可运行建模部分。

## 分析流程

### 一、数据理解

1. 加载 `numpy` / `pandas` / `matplotlib` / `seaborn`
2. 读取数据，删除 `date` 列（与房价无因果关系的记录时间）
3. 输出前 5 行、属性信息（`info`）、描述信息（`describe`）
4. 查看空值分布

**一个排查细节**：`sqft_above` 字段初步看有两个缺失值，怀疑是空格字符。用 `df['sqft_above'].dropna()` 验证后确认不是空格，而是真实的缺失值。

### 二、探索性分析（EDA）

5. 查看各属性与房价的分布关系
6. 绘制属性间相关系数热力图（`sns.heatmap`，剔除 `id` 列）
7. 用掩码（`mask`）只保留相关系数 ≥ 0.5 的强相关项，定位与房价强相关的核心特征
8. 绘制经纬度散点图，观察房屋的地理分布规律
9. 分析邮编（`zipcode`）与房价的分布关系

### 三、建模

10. `train_test_split` 划分训练集与测试集
11. `StandardScaler` 对特征做标准化，保存标准化后的数据

### 四、模型对比

| 模型 | 参数 | R² |
| --- | --- | --- |
| 决策树回归 | `DecisionTreeRegressor()` | 0.7778 |
| K 近邻回归 | `KNeighborsRegressor(n_neighbors=10)` | 0.8012 |
| 随机森林回归 | `RandomForestRegressor(n_estimators=500)` | 0.8834 |
| 梯度提升树 | `GradientBoostingRegressor(n_estimators=500)` | **0.8925** |

### 五、网格搜索调参

对梯度提升树用 `GridSearchCV` 搜索 `alpha`、`loss`、`min_samples_leaf` 三个超参数：

```
Best score is: 0.8387898399291445
Best parameter is: {'alpha': 0.9, 'loss': 'huber', 'min_samples_leaf': 4}
```

**注意**：网格搜索的最优得分（0.8388）低于单次划分的梯度提升树得分（0.8925）。原因是 GridSearchCV 的分数是交叉验证的平均值，比单次 train/test 划分的结果更保守，也更接近模型真实泛化能力。两者不可直接比较。

### 六、最终模型

用搜索到的最优参数重建模型：

```python
GradientBoostingRegressor(n_estimators=500, alpha=0.9, loss='huber', min_samples_leaf=4)
```

| 指标 | 值 |
| --- | --- |
| R² | **0.8862** |
| 平均绝对误差 MAE | 0.1859 |
| 均方根误差 RMSE | 0.1062 |

> 目标变量 `price` 已做标准化，因此 MAE / RMSE 数值较小，单位为标准化后的量纲。R² 不受量纲影响，可直接解读。

## 环境要求

```bash
pip install -r requirements.txt
```

## 运行

```bash
jupyter notebook house_price_forecast.ipynb
```

Notebook 已保存全部运行输出，可直接查看结果，无需重新运行。

## 目录结构

```
house-price-forecast-ml/
├── README.md
├── requirements.txt
├── .gitignore
└── house_price_forecast.ipynb    # 完整实验（110 个单元，含输出）
```

## 说明

- 课程大作业，按实验指导书给定的分析流程独立完成。
- `sns.heatmap` 的中文标签依赖系统中文字体。
- sklearn 训练时会输出 `DataConversionWarning`（传入列向量 y），不影响结果，把 `y` 用 `ravel()` 展平即可消除。
