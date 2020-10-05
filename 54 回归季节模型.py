#-*- coding: utf-8 -*-

########  本文件实现Holt-Winters季节预测模型，包括
#   Part1 基于回归的季节加法模型
#   Part2 基于回归的季节乘法模型

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

######################################################################
########  Part5、 基于回归的季节加法模型
######################################################################
# y(t) = base + trend*t + Seasional


# 1、读取数据集
filename = '时间序列.xls'
sheet = 'AirlineMiles'
df = pd.read_excel(filename, sheet)

# 2、预处理
df.rename(columns={'日期':'date','里程(万)':'y'},inplace=True)
df['t'] = range(1, len(df)+1)
df['month'] = df['date'].dt.month

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(dtype='int', sparse=False)
X_ = enc.fit_transform(df[['month']])
cols = []
for m in enc.categories_[0]:
    cols.append('m{}'.format(m))
dfMonth = pd.DataFrame(X_, index=df.index,columns=cols)


df = pd.concat([df[['y','t']], dfMonth], axis=1)


# 3、训练模型

from sklearn import metrics as mts
from scipy import optimize as spo

def optimizeParams(df, params, bnds=None, cons=() ):
    """
    基于回归的加法模型，寻找最优参数
    """
    y = df['y']
    X = df.iloc[:, 1:]

    # 定义误差函数（参数:数组，其它参数:元组）
    def error(params, *var):
        # X = var[0]
        # y = var[1]

        base = params[0]
        # trend = params[1]
        # months = params[2]
        p = params[1:]
        yhat = (base + X * p).sum(axis=1)
        
        mse = mts.mean_squared_error(y, yhat)
        return mse

    # 优化
    optResult = spo.minimize(error, params, args=(X,y),
                method='trust-constr',
                bounds=bnds, constraints=cons, 
                # options={'maxiter':1000,'disp':True}
                )
    print(optResult)

    return optResult['x']   #只返回最优参数

# 初始参数
params = [30,2]
params.extend([1]*12)

# 约束条件
# type='eq',表示fun值=0；type='ineq',表示fun值为非负数
cons = ({'type':'eq', 'fun': lambda params: params[2:].sum()})

# 定义base,trend,months的边界
months = [(-10,10)]*12  #定义月份的边界
bnds = [(30,50), (-5,5)]
bnds.extend(months)

bestParams = optimizeParams(df, params, bnds, cons)
p = np.round(bestParams, 2)
print('最优参数：', p)

# 4.评估模型
# 6.应用模型（略）

######################################################################
########  Part6 基于回归的季节乘法模型
######################################################################
# y(t) = base * trend^t * Seasional
