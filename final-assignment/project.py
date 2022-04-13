import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
# plt.rcParams['font.sans-serif']=['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  #用来正常显示负号

#读取并划分变量
df = pd.read_csv('happiness_train_abbr.csv')
# x_df = df.drop('happiness',axis = 1)
y_df = df['happiness']

sns.distplot(y_df, bins=15, kde=False)
y_df.value_counts()

df_clear = df.drop(df[df['happiness']<0].index)
sample_size = len(df_clear)
for col in df_clear.columns:
    nan_num = df_clear[col].isnull().sum()
    if pd.api.types.is_numeric_dtype(df_clear[col]):
        negative_num = (df_clear[col] < 0.01).sum()
        print(f'{col}: nan {nan_num}/{sample_size}, {nan_num/sample_size}, negative {negative_num}/{sample_size}, {negative_num/sample_size}')
    else:
        print(f'{col}: nan {nan_num}/{sample_size}, {nan_num/sample_size}')

numeric_columns = ['birth','religion_freq','edu','income','floor_area','height_cm','weight_jin','health','health_problem','depression','equity','class',
# 'work_status','work_yr','work_type','work_manage',
'family_income','family_m','family_status','house','car','status_peer','status_3_before','view']
classification_columns = ['survey_type',
'province','city','county',
'gender','nationality','religion','political','hukou','socialize','relax','learn','work_exper','marital','inc_ability']
drop_columns = ['survey_time',
# 'province' , 'city', 'county',
'work_status','work_yr','work_type','work_manage']
df.columns

for col in df_clear.columns:
    series = df_clear[col]
    if col in classification_columns:
        print(f'{col}: {len(series.unique())}')

for col in ['province', 'city', 'county']:
    fig, axis1 = plt.subplots(1,1,figsize=(12,6))
    class_num = pd.DataFrame({col:df_clear[col].values})
    sns.countplot(x=col, data=class_num, order=df_clear[col].value_counts().index, palette='ch:s=2.3,rot=.53,reverse=1')
    axis1.set_title(col)

df_clear = df_clear.drop(drop_columns, axis=1)
for col in df_clear.columns:
    series = df_clear[col]
    if col in numeric_columns:
        df_clear.loc[series.isnull(), col] = series.mean()
        df_clear.loc[series < 0.01, col] = series.mean()
    elif col in classification_columns:
        df_clear.loc[series.isnull(), col] = series.mode()[0]
        df_clear.loc[series < 0.01, col] = series.mode()[0]
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_clear.loc[:, numeric_columns] = scaler.fit_transform(df_clear.loc[:, numeric_columns])
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
embeddings = encoder.fit_transform(df_clear.loc[:, classification_columns])
df_clear = df_clear.drop(classification_columns, axis=1)
df_clear = pd.concat([df_clear, pd.DataFrame(embeddings)], axis=1)
df_clear = df_clear.dropna(axis=0)
df_clear.describe()

# 将数据分为训练集和测试集.

# 划分变量
y_df = df_clear['happiness']
x_df = df_clear.drop(['happiness','id'],axis = 1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x_df,y_df,test_size=0.2, random_state=123)

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

# 机器学习模型
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

# ## 模型融合
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
            j = 0
            for idx_train, idx_valid in kf.split(range(len(y))):
                x_train_j = X[idx_train]
                y_train_j = y[idx_train, :]
                x_valid_j = X[idx_valid]

                mod.fit(x_train_j, y_train_j.ravel())

                y_valid_j = mod.predict(x_valid_j)[:].reshape((-1, 1))
                s_train[idx_valid, :, i] = y_valid_j

                j += 1

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

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y).reshape((y.shape[0], -1))
        T = np.array(T)

        self.y_dim = y.shape[1]

        kf = KFold(n_splits=self.n_folds, shuffle=True,
                   random_state=self.rand_seed)

        s_train = np.zeros((X.shape[0], (len(self.base_models)), self.y_dim))
        s_test = np.zeros((T.shape[0], (len(self.base_models)), self.y_dim))
        y_predict = np.zeros((T.shape[0], (len(self.stackers)), self.y_dim))
        y_predict_weighted = np.zeros((T.shape[0], self.y_dim))

        for i, mod in enumerate(self.base_models):
            s_test_i = np.zeros((
                s_test.shape[0], kf.get_n_splits(), self.y_dim))

            j = 0
            for idx_train, idx_valid in kf.split(range(len(y))):
                x_train_j = X[idx_train]
                y_train_j = y[idx_train, :]
                x_valid_j = X[idx_valid]

                mod.fit(x_train_j, y_train_j.ravel())

                y_valid_j = mod.predict(x_valid_j).reshape((-1, 1))
                s_train[idx_valid, i, :] = y_valid_j
                s_test_i[:, j, :] = mod.predict(T).reshape((-1, 1))

                j += 1

            s_test[:, i, :] = s_test_i.mean(1)

        for stacker in self.stackers:
            stacker.fit(s_train.reshape((s_train.shape[0], -1)), y.ravel())

        for i, stacker in enumerate(self.stackers):
            temp_y_predict = stacker.predict(
                s_test.reshape((s_test.shape[0], -1)))
            y_predict[:, i, :] = temp_y_predict[:].reshape((-1, 1))
            y_predict_weighted += y_predict[:, i, :] * self.weights[i]

        stackers_num = len(self.stackers)
        for i, result in enumerate(self.added_results):
            y_predict_weighted += result * self.weights[stackers_num + i]

        return y_predict_weighted

stacking = Stacking(n_folds=5, rand_seed=123, base_models=model_list, stackers=[
    LogisticRegression(C=0.1, max_iter=1000, random_state=123)
    ], weights=[1.0])

stacking.fit(X_train,y_train)
y_predict = stacking.predict(X_test)
score = mean_absolute_error(y_predict,y_test)
print(f'Stacking: {score}')


