#-*- coding: utf-8 -*-

########  本文件实现Holt趋势法（亦称二次指数平滑），包括
#   Part1 霍尔特线性趋势法:Holt’s Linear trend method
#   Part2 霍尔特指数趋势法:Holt’s Exponential trend method
#   Part3 霍尔特阻尼线性趋势法:Holt’s Damped linear trend method
#   Part4 霍尔特阻尼指数趋势法:Holt’s Damped Exponential trend method

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
def displayRegressionMetrics(y_true, y_pred, adjVal=None):
    '''
    \n功能：计算回归的各种评估指标。
    \n参数：y_true:真实值
         y_pred:预测值
         adjVal:输入的shape参数(n,p)，其中n是样本量，p是特征数
            默认None表示是一元回归；
    \n返回：各种指标，字典形式
    '''
    # 评估指标：R^2/adjR^2, MAPE, MAE，RMSE
    mts = {}
    #一元回归，计算R^2；
    mts['R2'] = metrics.r2_score(y_true, y_pred)
    # 多元回归，计算调整R^2
    if (adjVal != None) and (adjVal[1] > 1):
        n, p = adjVal
        mts['adjR2']  = 1-(1-mts['R2'])*(n-1)/(n-p-1)

    mts['MAPE'] = (abs((y_pred-y_true)/y_true)).mean()
    mts['MAE'] = metrics.mean_absolute_error(y_true, y_pred)
    MSE = metrics.mean_squared_error(y_true, y_pred)
    mts['RMSE'] = np.sqrt(MSE)
    
    # 格式化，保留小数点后4位
    for k,v in mts.items():
        mts[k] = np.round(v, 4)
    
    # 特别处理,注意变成了字符串
    mts['MAPE'] = '{0:.2%}'.format(mts['MAPE']) 

    # # 残差检验：均值为0，正态分布，随机无自相关
    # resid = y_true - y_pred         #残差
    # z,p = stats.normaltest(resid)   #正态检验
    
    print(mts)
    return None


# 1、读取数据集
filename = '时间序列.xls'
sheet = '餐厅销量'
df = pd.read_excel(filename, sheet)
print(df.columns.tolist())

df.set_index('日期', inplace=True)
ts = df['销量']
ts.index.freq = 'D'

# 2、可视化
ts.plot(title='序列')
plt.show()

from statsmodels.tsa.holtwinters import Holt
# Holt(endog, exponential=False, damped=False)
    # ############ Holt趋势种类 ###################
    # Holt’s Linear trend method             (damped=False, exponential=False)
    # Holt’s Exponential trend method        (damped=False, exponential=True)
    # Holt’s Damped linear trend method      (damped=True, exponential=False)
    # Holt’s Damped Exponential trend method (damped=True, exponential=True)

# Holt.fit(
    # smoothing_level=None, 
    # smoothing_slope=None, 
    # damping_slope=None, 
    # optimized=True,  默认自行优化未指定的参数alpha,beta等
    # start_params=None, 
    # initial_level=None, 
    # initial_slope=None, 
    # use_brute=True)

######################################################################
########  Part1、霍尔特线性趋势法:Holt’s Linear trend method
######################################################################

# 3.训练模型
alpha = 0.8
beta = 0.2

# 线性：指定固定的alpha, beta
mdl = Holt(ts)
results = mdl.fit(
                smoothing_level=alpha, 
                smoothing_slope=beta,
                optimized=False)

# results = Holt(ts).fit(
#                 smoothing_level=alpha, 
#                 smoothing_slope=beta,
#                 optimized=False)
# print(results.summary())

print('参数：\n', results.params)

# 4.模型评估
print(results.aic, results.aicc, results.bic)
# print(results.resid)    #残差resid=y - pred
y_pred = results.fittedvalues
y_true = ts

displayRegressionMetrics(y_true, y_pred)

df = pd.concat([y_true, y_pred], axis=1)
# print(df)

# 可视化图形
plt.plot(range(len(y_pred)), y_pred, 'b', label='Holt指数平滑')
plt.plot(range(len(y_true)), y_true, 'g', label='实际值')

plt.legend(loc='upper right')
title = 'Holt线性趋势(alpha={},beta={})'.format(alpha,beta)
plt.title(title)
plt.show()

# 5.模型优化:自行寻找最优参数
results = Holt(ts).fit()    #optimized=True
# print(results.summary())
print(results.params)

best_alpha = results.params['smoothing_level']
best_beta = results.params['smoothing_slope']

print('最优水平平滑因子：\n', best_alpha)
print('最优趋势平滑因子：\n', best_beta)

# 其余类似:评估、可视化等等

# 6.应用模型
# 1)预测历史值
pred = results.predict(start='2015-02-01', end='2015-02-07')
print(pred)
# 2)预测未来值(滚动预测)
pred = results.forecast(5)
print(pred)

# 3)保存模型
fname = 'out.pkl'
results.save(fname)

# 4）加载模型
from statsmodels.iolib.smpickle import load_pickle
results = load_pickle(fname)

# 5)应用模型
print(results.params)
pred = results.forecast(3)
print(pred)

######################################################################
########  Part2、霍尔特指数趋势法:Holt’s Exponential trend method
######################################################################

# 3.训练模型（自动优化参数alpha,beta）
results = Holt(ts, exponential=True).fit()
# print(results.summary())
print(results.params)

best_alpha = results.params['smoothing_level']
best_beta = results.params['smoothing_slope']

print('最优水平平滑因子：\n', best_alpha)
print('最优趋势平滑因子：\n', best_beta)

# 4.评估（略）
y_pred = results.fittedvalues
y_true = ts
displayRegressionMetrics(y_true, y_pred)

# 6.应用模型（略）


######################################################################
########  Part3、霍尔特阻尼线性趋势法:Holt’s Damped linear trend method
######################################################################

# 3、训练模型
results = Holt(ts, damped=True).fit()
# print(results.summary())
print(results.params)

best_alpha = results.params['smoothing_level']
best_beta = results.params['smoothing_slope']
best_phi = results.params['damping_slope']

# 4.评估模型
y_pred = results.fittedvalues
y_true = ts
displayRegressionMetrics(y_true, y_pred)

# 6.应用模型（略）



######################################################################
########  Part4 霍尔特阻尼指数趋势法:Holt’s Damped Exponential trend method
######################################################################


# 3、训练模型
results = Holt(ts, damped=True, exponential=True).fit()
# print(results.summary())
print(results.params)

best_alpha = results.params['smoothing_level']
best_beta = results.params['smoothing_slope']
best_phi = results.params['damping_slope']

# 4.评估模型
y_pred = results.fittedvalues
y_true = ts
displayRegressionMetrics(y_true, y_pred)

# 6.应用模型（略）



# ##############最优模型选择, 选择条件：
# 1)参数不能为边界值（如0或1）
# 2)AIC/BIC：信息损失越小越好
# 3)R^2：拟合程度越大越好
# 4)MAE/MAPE/RMSE：模型误差越小越好
idxs = ['smoothing_level','smoothing_slope','damping_slope']
clsParams = [('Holt线性',{'damped':False, 'exponential':False}),
            ('Holt指数', {'damped':False, 'exponential':True}),
            ('Holt阻尼线性', {'damped':True, 'exponential':False}),
            ('Holt阻尼指数', {'damped':True, 'exponential':True})]
for name, params in clsParams:
    print('\n模型：',name)
    results = Holt(endog=ts, **params).fit()
    for idx in idxs:
        print(idx, '=', results.params[idx])
    print('AIC=', results.aic)
    y_pred = results.fittedvalues
    displayRegressionMetrics(ts, y_pred)

# 可知