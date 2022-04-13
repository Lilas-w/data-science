import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  #用来正常显示负号

#读取并划分变量
df = pd.read_csv('breast_cancer.csv')
y_df = df['type']
x_df = df.drop('type',axis = 1)

y_df.value_counts()

pie_plot = y_df.value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',shadow=True,figsize=(3,3))
pie_plot.set_title('标签比例')
pie_plot.set_ylabel('')
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x_df,y_df,test_size=0.2, random_state=123)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=40,random_state=123)
model.fit(X_train,y_train)

y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_predict,y_test)
score

from sklearn.metrics import confusion_matrix
i = pd.DataFrame(confusion_matrix(y_test, y_predict),index = ['实际不患癌症','实际患癌症'],columns = ['预测不患癌症','预测患癌症'])
i

y_pred_proba = model.predict_proba(X_test)  #f(y)
y_pred_proba
a = pd.DataFrame(y_pred_proba,columns = ['分类为0的概率','分类为1的概率'])
a

color1 = '#FF0000'
color2 = '#0000FF'
plt.scatter(y_predict[y_test==1],y_pred_proba[:,1][y_test==1],c=color1, alpha=0.4, label='正样本')
plt.scatter(y_predict[y_test==0],y_pred_proba[:,1][y_test==0],c=color2, alpha=0.4, label='负样本')
plt.show()

# 绘制ROC曲线
from sklearn.metrics import roc_curve

fpr, tpr, thres = roc_curve(y_test.values, y_pred_proba[:,1])
plt.plot(fpr, tpr)
plt.show()

# 查看AUC值
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test, y_pred_proba[:,1])
print(score)

lst = [1,5,10,20,30,40,50,60,70,80,90,100,200,500,1000]
score_list = []
for i in lst:
    model = RandomForestClassifier(max_depth=1, n_estimators=i, random_state=123)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    score = accuracy_score(y_predict,y_test)
    score_list.append(score)
    print(i,score)    

plt.plot(lst, score_list)
plt.show()