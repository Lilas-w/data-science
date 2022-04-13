import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  #用来正常显示负号

#读取并划分变量
df = pd.read_csv('happiness_train_abbr.csv')
y_df = df['happiness']
x_df = df.drop('happiness',axis = 1)

print(y_df.value_counts())

for col in x_df.columns:
    print(col)
    print(x_df[col].isnull().sum())

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

model = LinearRegression()