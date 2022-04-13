
# python大数据分析期末作业——幸福感预测

[TOC]


幸福感是一个古老而深刻的话题，是人类世代追求的方向。与幸福感相关的因素成千上万、因人而异，大如国计民生，小如路边烤红薯，都会对幸福感产生影响。本次作业使用调查数据对人们的幸福感进行预测，赛题来自于天池竞赛：[快来一起挖掘幸福感！相关的问题-天池大赛-阿里云天池 (aliyun.com)](https://tianchi.aliyun.com/competition/entrance/231702/introduction)。

## 数据集探索
调查数据来自中国人民大学中国调查与数据中心主持的《中国综合社会调查（CGSS）》项目。中国综合社会调查为多阶分层抽样的截面面访调查。调查数据包含多组变量，包括个体变量（性别、年龄、地域、职业、健康、婚姻与政治面貌等等）、家庭变量（父母、配偶、子女、家庭资本等等）、社会态度（公平、信用、公共服务等等），可用于预测其对幸福感的评价。

首先加载数据集，并对需要预测的幸福感指标的数据分布进行可视化分析：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
# plt.rcParams['font.sans-serif']=['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  #用来正常显示负号

df = pd.read_csv('happiness_train_abbr.csv')
sns.distplot(df['happiness'], bins=15, kde=False)
df['happiness'].value_counts()
```


     4    4818
     5    1410
     3    1159
     2     497
     1     104
    -8      12
    Name: happiness, dtype: int64


![png](https://gitee.com/ldx_pku/image/raw/master/img/output_1_1.png)



由图和统计数据可知，在8000个幸福感样例数据中，存在12个无效数据。在余下的有效数据中，幸福感分为1-5共5个等级，半数以上的人评价自己的幸福感为第4档。

接下来去除无效数据，

```python
df_clear = df.drop(df[df['happiness']<0].index)
```

并对特征数据是否完整进行分析：

```python
sample_size = len(df_clear)
for col in df_clear.columns:
    nan_num = df_clear[col].isnull().sum()
    if pd.api.types.is_numeric_dtype(df_clear[col]):
        negative_num = (df_clear[col] < 0.01).sum()
        print(f'{col}: nan {nan_num}/{sample_size}, {nan_num/sample_size}, negative {negative_num}/{sample_size}, {negative_num/sample_size}')
    else:
        print(f'{col}: nan {nan_num}/{sample_size}, {nan_num/sample_size}')
```

    id: nan 0/7988, 0.0, negative 0/7988, 0.0
    happiness: nan 0/7988, 0.0, negative 0/7988, 0.0
    survey_type: nan 0/7988, 0.0, negative 0/7988, 0.0
    province: nan 0/7988, 0.0, negative 0/7988, 0.0
    city: nan 0/7988, 0.0, negative 0/7988, 0.0
    county: nan 0/7988, 0.0, negative 0/7988, 0.0
    survey_time: nan 0/7988, 0.0
    gender: nan 0/7988, 0.0, negative 0/7988, 0.0
    birth: nan 0/7988, 0.0, negative 0/7988, 0.0
    nationality: nan 0/7988, 0.0, negative 18/7988, 0.002253380070105158
    religion: nan 0/7988, 0.0, negative 950/7988, 0.11892839258888332
    religion_freq: nan 0/7988, 0.0, negative 15/7988, 0.0018778167250876315
    edu: nan 0/7988, 0.0, negative 9/7988, 0.001126690035052579
    income: nan 0/7988, 0.0, negative 1632/7988, 0.20430645968953431
    political: nan 0/7988, 0.0, negative 39/7988, 0.0048823234852278415
    floor_area: nan 0/7988, 0.0, negative 0/7988, 0.0
    height_cm: nan 0/7988, 0.0, negative 0/7988, 0.0
    weight_jin: nan 0/7988, 0.0, negative 0/7988, 0.0
    health: nan 0/7988, 0.0, negative 3/7988, 0.0003755633450175263
    health_problem: nan 0/7988, 0.0, negative 41/7988, 0.005132699048572859
    depression: nan 0/7988, 0.0, negative 14/7988, 0.0017526289434151227
    hukou: nan 0/7988, 0.0, negative 0/7988, 0.0
    socialize: nan 0/7988, 0.0, negative 4/7988, 0.000500751126690035
    relax: nan 0/7988, 0.0, negative 15/7988, 0.0018778167250876315
    learn: nan 0/7988, 0.0, negative 19/7988, 0.0023785678517776665
    equity: nan 0/7988, 0.0, negative 39/7988, 0.0048823234852278415
    class: nan 0/7988, 0.0, negative 76/7988, 0.009514271407110666
    work_exper: nan 0/7988, 0.0, negative 0/7988, 0.0
    work_status: nan 5042/7988, 0.6311967951927892, negative 23/7988, 0.0028793189784677015
    work_yr: nan 5042/7988, 0.6311967951927892, negative 162/7988, 0.02028042063094642
    work_type: nan 5042/7988, 0.6311967951927892, negative 66/7988, 0.008262393590385579
    work_manage: nan 5042/7988, 0.6311967951927892, negative 57/7988, 0.007135703555332999
    family_income: nan 1/7988, 0.00012518778167250875, negative 780/7988, 0.09764646970455683
    family_m: nan 0/7988, 0.0, negative 21/7988, 0.002628943415122684
    family_status: nan 0/7988, 0.0, negative 42/7988, 0.005257886830245368
    house: nan 0/7988, 0.0, negative 625/7988, 0.07824236354531798
    car: nan 0/7988, 0.0, negative 8/7988, 0.00100150225338007
    marital: nan 0/7988, 0.0, negative 0/7988, 0.0
    status_peer: nan 0/7988, 0.0, negative 46/7988, 0.005758637956935403
    status_3_before: nan 0/7988, 0.0, negative 45/7988, 0.005633450175262894
    view: nan 0/7988, 0.0, negative 203/7988, 0.02541311967951928
    inc_ability: nan 0/7988, 0.0, negative 959/7988, 0.12005508262393591

工作相关的四个特征（work_status, work_yr, work_type, work_manage）缺失严重，考虑将其舍去。其余特征信息均相对完整，但有一些数值特征存在不合常理的负值，如heath_problem健康问题的数量等，需进一步替换补全。

可将全部特征（共计40个）分为数值特征和分类特征两类：

```python
numeric_columns = ['birth','religion_freq','edu','income','floor_area','height_cm','weight_jin','health','health_problem','depression','equity','class',
# 'work_status','work_yr','work_type','work_manage',
'family_income','family_m','family_status','house','car','status_peer','status_3_before','view']
classification_columns = ['survey_type',
'province','city','county',
'gender','nationality','religion','political','hukou','socialize','relax','learn','work_exper','marital','inc_ability']
drop_columns = ['survey_time',
# 'province' , 'city', 'county',
'work_status','work_yr','work_type','work_manage']
df_clear.columns
```


    Index(['id', 'survey_type', 'province', 'city', 'county', 'survey_time',
           'gender', 'birth', 'nationality', 'religion', 'religion_freq', 'edu',
           'income', 'political', 'floor_area', 'height_cm', 'weight_jin',
           'health', 'health_problem', 'depression', 'hukou', 'socialize', 'relax',
           'learn', 'equity', 'class', 'work_exper', 'work_status', 'work_yr',
           'work_type', 'work_manage', 'family_income', 'family_m',
           'family_status', 'house', 'car', 'marital', 'status_peer',
           'status_3_before', 'view', 'inc_ability'],
          dtype='object')

使用分类特征进行预测时，需要将每一个分类进行embedding，各种embedding方案中One Hot方式效果较好，但如果分类过多会使得embedding维数过大。打印全部分类特征的分类数：


```python
for col in df_clear.columns:
    series = df_clear[col]
    if col in classification_columns:
        print(f'{col}: {len(series.unique())}')
```

    survey_type: 2
    province: 28
    city: 85
    county: 130
    gender: 2
    nationality: 9
    religion: 3
    political: 5
    hukou: 8
    socialize: 6
    relax: 6
    learn: 6
    work_exper: 6
    marital: 7
    inc_ability: 5

可以看出，除了三个与地区相关的特征（省、市、县）相关外，其余特征的分类数均不高，可以直接进行One Hot embedding。对于这三个分类数较多的特征，考虑将频数较低的尾部分类合并。将所有分类的频数排序并绘图展示：

```python
for col in ['province', 'city', 'county']:
    fig, axis1 = plt.subplots(1,1,figsize=(12,6))
    class_num = pd.DataFrame({col:df_clear[col].values})
    sns.countplot(x=col, data=class_num, order=df_clear[col].value_counts().index)
    axis1.set_title(col)
```


![png](https://gitee.com/ldx_pku/image/raw/master/img/output_5_0.png)

![png](https://gitee.com/ldx_pku/image/raw/master/img/output_5_1.png)

![png](https://gitee.com/ldx_pku/image/raw/master/img/output_5_2.png)



从图中可以看到，三类数据的长尾现象并不明显，类别合并难以进行。考虑到将全部分类特征进行One Hot Embedding的总特征维数为300左右，仍然是可以接受的，因此决定不再合并类别。

## 数据预处理

首先将不需要的特征舍去：

```python
df_clear = df_clear.drop(drop_columns, axis=1)
```

根据之前的数据探索，特征数据在用于训练和预测前需要先补全和调整。对于数值特征，使用均值填补缺失值和负值；对于分类特征，则使用众数填补。

```python
for col in df_clear.columns:
    series = df_clear[col]
    if col in numeric_columns:
        df_clear.loc[series.isnull(), col] = series.mean()
        df_clear.loc[series < 0.01, col] = series.mean()
    elif col in classification_columns:
        df_clear.loc[series.isnull(), col] = series.mode()[0]
        df_clear.loc[series < 0.01, col] = series.mode()[0]
```

在用模型预测和分类前，对特征实行标准化操作，可使模型性能更加稳定。对数值特征使用StandardScaler标准化操作，将数值放缩至[0-1]的范围内。

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_clear.loc[:, numeric_columns] = scaler.fit_transform(df_clear.loc[:, numeric_columns])
```

分类特征全部使用One Hot embedding处理，最终生成的特征向量全部维数为320维：

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
embeddings = encoder.fit_transform(df_clear.loc[:, classification_columns])
df_clear = df_clear.drop(classification_columns, axis=1)
df_clear = pd.concat([df_clear, pd.DataFrame(embeddings)], axis=1)
df_clear = df_clear.dropna(axis=0)
df_clear.describe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>happiness</th>
      <th>birth</th>
      <th>religion_freq</th>
      <th>edu</th>
      <th>income</th>
      <th>floor_area</th>
      <th>...</th>
      <th>299</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7976.000000</td>
      <td>7976.000000</td>
      <td>7976.000000</td>
      <td>7976.000000</td>
      <td>7976.000000</td>
      <td>7976.000000</td>
      <td>7976.000000</td>
      <td>...</td>
      <td>7976.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3995.596540</td>
      <td>3.868104</td>
      <td>0.574973</td>
      <td>0.055673</td>
      <td>0.299619</td>
      <td>0.004010</td>
      <td>0.086542</td>
      <td>...</td>
      <td>0.026956</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2306.398885</td>
      <td>0.818841</td>
      <td>0.221681</td>
      <td>0.168646</td>
      <td>0.239948</td>
      <td>0.023274</td>
      <td>0.067180</td>
      <td>...</td>
      <td>0.161965</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1998.750000</td>
      <td>4.000000</td>
      <td>0.407895</td>
      <td>0.000000</td>
      <td>0.153846</td>
      <td>0.001075</td>
      <td>0.047032</td>
      <td>...</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3997.500000</td>
      <td>4.000000</td>
      <td>0.578947</td>
      <td>0.000000</td>
      <td>0.230769</td>
      <td>0.002995</td>
      <td>0.071704</td>
      <td>...</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5993.250000</td>
      <td>4.000000</td>
      <td>0.736842</td>
      <td>0.000000</td>
      <td>0.384615</td>
      <td>0.003595</td>
      <td>0.097918</td>
      <td>...</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7988.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 322 columns</p>
在训练和评测前将数据分为训练集和测试集：


```python
# 划分变量
y_df = df_clear['happiness']
x_df = df_clear.drop(['happiness','id'], axis = 1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=123)
```

## 机器学习模型

为探索本数据集对于机器学习模型类型的适用性，此次预测任务选择了多种类型的回归模型，分别为：逻辑回归模型(LinearRegression)，基于线性的线性回归(LinearRegression)、岭回归(Ridge)、ElasticNet模型，支持向量机模型(SVR)，基于树与boosting的GradientBoostingRegressor、随机森林模型(RandomForestRegressor)和Light-GBM(LGBMRegressor)。


```python
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

model_list = []
model_name_list = []
score_list = []

model_list.append(LogisticRegression(solver='newton-cg', C=0.1, max_iter=10000, random_state=123))
# model_list.append(LinearRegression())
model_list.append(Ridge(alpha=2000.0, max_iter=100000, random_state=123))
model_list.append(ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=10000, random_state=123))
model_list.append(SVR(C=0.01, gamma=0.1))
model_list.append(GradientBoostingRegressor(loss='huber', max_features='sqrt', n_estimators=500, random_state=123))
model_list.append(RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_split=3, random_state=123))
model_list.append(LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.01, random_state=123))


for model in model_list:
    raw_cls_name = str(type(model))
    model_name = raw_cls_name[raw_cls_name.rfind('.') + 1:raw_cls_name.rfind("'")]
    model_name_list.append(model_name)

for model, model_name in zip(model_list, model_name_list):
    model.fit(X_train,y_train)
    y_predict = model.predict(X_test)
    score = mean_absolute_error(y_predict,y_test)
    score_list.append(score)
    print(f'{model_name}: {score}')
```

    LogisticRegression: 0.4448621553884712
    Ridge: 0.5069413808178973
    ElasticNet: 0.5041977278438632
    SVR: 0.4732277487104374
    GradientBoostingRegressor: 0.5041471065425246
    RandomForestRegressor: 0.49322739446535835
    LGBMRegressor: 0.4845034240067739

除简单线性回归不能在本数据集上达到收敛外，所有模型在使用GridSearchCV调整参数后，MAE都达到了接近或低于0.50的水平。表现最佳的是逻辑回归模型，0.44的MAE远好于其他linear-based和tree-based模型，SVM模型、Light-GBM模型为次佳。

## 模型融合

由于没有做过多的特征交叉处理，单一的统计回归模型达到的效果有限，故采用模型融合的方式提升表现效果。使用Stacking，将基础模型通过贝叶斯优化(Bayesian Optimization)的方法在训练集上确定最佳参数，得到融合后的模型。整个Stacking的过程类似于CV验证：

- 将训练集分为五份，对每个基本模型进行5轮训练，一次使用其中的4份作为训练集训练，预测余下一份的结果，5轮后得到训练集大小的预测数据，同时在每轮中对测试集进行预测，对每个基本模型来说，测试集的预测结果为5轮结果的均值。
- 在第二层中，输⼊大小为训练集上的$预测结果*基本模型数量$的数据进行训练。
- 第三层最终输出为第二层的预测结果。

![image-20200620214210165](https://gitee.com/ldx_pku/image/raw/master/img/image-20200620214210165.png)

在基础模型的选择上，尽量选择不同种类的模型，并且只选择表现较好的模型。Stacking过程相当于在n个基础模型上进行K-Fold的CV验证，因此，出于效率考虑，一些训练太慢的模型也不宜作为基础模型。故编写以下Stacking类用于模型融合，融合模型的stacker选择使用简单且效果好的LogisticRegressiong：


```python
class Stacking(object):
    def __init__(self, n_folds, rand_seed, base_models, stackers, weights):
        assert np.array(weights).sum() == 1.0
        self.y_dim = 1
        self.n_folds = n_folds
        self.rand_seed = rand_seed
        self.base_models = base_models
        self.stackers = stackers
        self.weights = weights

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape((y.shape[0], -1))

        self.y_dim = y.shape[1]

        kf = KFold(n_splits=self.n_folds, shuffle=True,
                   random_state=self.rand_seed)

        s_train = np.zeros((X.shape[0], self.y_dim, len(self.base_models)))

        for i, mod in enumerate(self.base_models):
            for idx_train, idx_valid in kf.split(range(len(y))):
                x_train_j = X[idx_train]
                y_train_j = y[idx_train, :]
                x_valid_j = X[idx_valid]

                mod.fit(x_train_j, y_train_j.ravel())

                y_valid_j = mod.predict(x_valid_j)[:].reshape((-1, 1))
                s_train[idx_valid, :, i] = y_valid_j

        for stacker in self.stackers:
            stacker.fit(s_train.reshape(s_train.shape[0], -1), y.ravel())

    def predict(self, T):
        T = np.array(T)
        s_test = np.zeros((T.shape[0], self.y_dim, len(self.base_models)))
        y_predict = np.zeros((T.shape[0], self.y_dim, len(self.stackers)))
        y_predict_weighted = np.zeros((T.shape[0], self.y_dim))

        for i, mod in enumerate(self.base_models):
            s_test[:, :, i] = mod.predict(T).reshape((-1, 1))

        for i, stacker in enumerate(self.stackers):
            tmp_y_predict = stacker.predict(s_test.reshape((s_test.shape[0], -1)))
            y_predict[:, :, i] = np.array(tmp_y_predict).reshape((-1,1))

            y_predict_weighted += y_predict[:, :, i].reshape(y_predict_weighted.shape) * self.weights[i]

        return y_predict_weighted

stacking = Stacking(n_folds=5, rand_seed=123, base_models=model_list, stackers=[
    LogisticRegression(C=0.1, max_iter=1000, random_state=123)
    ], weights=[1.0])

stacking.fit(X_train,y_train)
y_predict = stacking.predict(X_test)
score = mean_absolute_error(y_predict,y_test)
print(f'Stacking: {score}')
```

    Stacking: 0.4342105263157895

模型融合后，最终MAE分数又有所改善，降至0.434，在所有结果中最低。该（验证集）分数在天池竞赛中约排至第6，说明融合模型可以较好地预测居民对幸福感的评价。

