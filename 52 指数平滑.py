#-*- coding: utf-8 -*-

########  本文件实现移动平均，包括
#   Part1 一次指数平滑/简单指数平滑
#   Part2 二次指数平滑
#   Part3 三次指数平滑

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

from common import displayRegressionMetrics


# 1、读取数据集
filename = '时间序列.xls'
sheet = '餐厅销量'
df = pd.read_excel(filename, sheet)
print(df.columns.tolist())

df.set_index('日期', inplace=True)
ts = df['销量']

# 2、可视化
ts.plot(title='餐厅销量')
plt.show()

######################################################################
########  Part1、一次指数平滑 Simple Exponential Smoothing
######################################################################
# SimpleExpSmoothing(endog)
# SimpleExpSmoothing.fit(smoothing_level=None, 
                # optimized=True,   默认优化未指定的参数
                # start_params=None, 
                # initial_level=None, 指定预测的初始值
                # use_brute=True)     默认使用暴力寻找初始值

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# 3.训练模型

# #######指定固定平滑系数
alpha = 0.8
mdl = SimpleExpSmoothing(ts)
results = mdl.fit(smoothing_level=alpha,
            optimized=False)
print(results.summary())    #打印模型信息

print('模型参数：\n', results.params)   #返回参数字典

# 4.模型评估
# 1）查看评估指数
print('AIC=', results.aic)
print('AICC=', results.aicc)
print('BIC=', results.bic)
# print('SSE=', results.sse)
# resid = results.resid     # 残差resid = ts - y_pred
y_pred = results.fittedvalues   # 预测历史值
y_true = ts[y_pred.index]

displayRegressionMetrics(y_true, y_pred)

# 2）可视化图形：对比
plt.plot(range(len(y_pred)), y_pred, 'b', label='一次指数平滑')
plt.plot(range(len(y_true)), y_true, 'g', label='实际值')

plt.legend(loc='upper right')
title = 'Holt线性趋势(alpha={})'.format(alpha)
plt.title(title)
plt.show()

# 5.自行选取最优平滑系数
mdl = SimpleExpSmoothing(ts)
results = mdl.fit()         #默认optimized=True
print(results.summary())    #打印模型信息

print('最优alpha=', results.params['smoothing_level'])
# 可惜已经取到了边界值，不合适

# 其余同上：指标、可视化
y_pred = results.fittedvalues   # 预测历史值
y_true = ts[y_pred.index]
displayRegressionMetrics(y_true, y_pred)

# 6.应用模型
# 1）预测历史值，end默认为最后日期
# 如果指定到未来的时期，将会进行滚动预测
pred = results.predict(start='2015-02-01', end='2015-02-10')
print(pred)

# 2）进行滚动预测，预测未来几个值
pred = results.forecast(5)
print(pred)
# 由于采用滚动预测，越远越不准确

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
########  Part2、二次指数平滑 Double Exponential Smoothing
######################################################################
# Brown's Linear Exponential Smoothing

def double_exponential_smoothing(ts, alpha=0.8, isPlot=True):
    """
    布朗线性趋势模型（二次指数平滑）
    """
    S1 = ts.ewm(alpha=alpha).mean()
    S2 = S1.ewm(alpha=alpha).mean()

    at = 2*S1 - S2
    bt = alpha*(S1 - S2)/(1-alpha)

    yhat = at + bt
    # ret = yhat[-1]
    pred = yhat.shift(1)
    y_pred = pred.dropna()
    y_true = ts[pred.index]

    # 计算误差率
    mape = (np.abs(y_pred - y_true)/y_true).mean()

    if isPlot:
        # 可视化图形
        plt.plot(range(len(y_pred)), y_pred, 'b', label='指数平滑')
        plt.plot(range(len(y_true)), y_true, 'g', label='实际值')
        
        plt.legend(loc='upper right')
        title = '二次指数平滑(alpha={})'.format(alpha)
        plt.title(title)
        plt.show()

    return mape

alpha = 0.5
mape = double_exponential_smoothing(ts, alpha)
print('MAPE={:.4%}'.format(mape))

# 5.寻找最优alpha
import scipy.optimize as spo


def optimizeDES(ts, alpha):
    """
    二次指数平滑，寻找最优平滑系数alpha
    """

    # 定义误差函数（参数:平滑系数，其它参数:元组）
    def error(alpha, *var):
        ts = var[0]
        S1 = ts.ewm(alpha=alpha).mean()
        S2 = S1.ewm(alpha=alpha).mean()

        at = 2*S1 - S2
        bt = alpha*(S1 - S2)/(1-alpha)

        yhat = at + bt
        # ret = yhat[-1]
        pred = yhat.shift(1)
        y_pred = pred.dropna()
        y_true = ts[y_pred.index]

        mse = metrics.mean_squared_error(y_true, y_pred)
        return mse

    # 约束条件
    bnds = [(0,1)]   #alpha取值范围

    # 优化
    optResult = spo.minimize(error, alpha, args=(ts,),
                method='SLSQP',
                bounds=bnds,
                # options={'maxiter':1000,'disp':True}
                )
    assert(optResult['success'])
    # print(optResult)

    return optResult['x']   #只返回最优参数

alpha = 0.5
best_alpha = optimizeDES(ts, alpha)
print('最优参数：', best_alpha)

mape = double_exponential_smoothing(ts, best_alpha)
print('MAPE={:.4%}'.format(mape))

######################################################################
########  Part3、三次指数平滑 Triple Exponential Smoothing
######################################################################


def Triple_exponential_smoothing(ts, alpha=0.8, isPlot=True):
    """
    三次指数平滑，适用于二次曲线趋势
    """
    S1 = ts.ewm(alpha=alpha).mean()
    S2 = S1.ewm(alpha=alpha).mean()
    S3 = S2.ewm(alpha=alpha).mean()

    at = 3*S1 - 3*S2 + S3
    bt = alpha*((6-5*alpha)*S1 - 2*(5-4*alpha)*S2 + (4-3*alpha)*S3)/(2*((1-alpha)**2))
    ct = (alpha**2)*(S1 - 2*S2 + S3)/(2*((1-alpha)**2))

    yhat = at + bt + ct
    # ret = yhat[-1]
    pred = yhat.shift(1)
    y_pred = pred.dropna()
    y_true = ts[pred.index]

    # 计算误差率
    mape = (np.abs(y_pred - y_true)/y_true).mean()

    if isPlot:
        # 可视化图形
        plt.plot(range(len(y_pred)), y_pred, 'b', label='指数平滑')
        plt.plot(range(len(y_true)), y_true, 'g', label='实际值')
        
        plt.legend(loc='upper right')
        title = '三次指数平滑(alpha={})'.format(alpha)
        plt.title(title)
        plt.show()

    return mape

alpha = 0.8
mape = Triple_exponential_smoothing(ts, alpha)
print('MAPE={:.4%}'.format(mape))


def optimizeTES(ts, alpha):
    """
    二次指数平滑，寻找最优平滑系数alpha
    """

    # 定义误差函数（参数:平滑系数，其它参数:元组）
    def error(alpha, *var):
        ts = var[0]
        S1 = ts.ewm(alpha=alpha).mean()
        S2 = S1.ewm(alpha=alpha).mean()
        S3 = S2.ewm(alpha=alpha).mean()

        at = 3*S1 - 3*S2 + S3
        bt = alpha*((6-5*alpha)*S1 - 2*(5-4*alpha)*S2 + (4-3*alpha)*S3)/(2*((1-alpha)**2))
        ct = (alpha**2)*(S1 - 2*S2 + S3)/(2*((1-alpha)**2))

        yhat = at + bt + ct
        # ret = yhat[-1]
        pred = yhat.shift(1)
        y_pred = pred.dropna()
        y_true = ts[y_pred.index]

        mse = metrics.mean_squared_error(y_true, y_pred)
        return mse

    # 约束条件
    bnds = [(0,1)]   #alpha取值范围

    # 优化
    optResult = spo.minimize(error, alpha, args=(ts,),
                method='SLSQP',
                bounds=bnds,
                # options={'maxiter':1000,'disp':True}
                )
    assert(optResult['success'])
    print(optResult)

    return optResult['x']   #只返回最优参数列表

alpha = 0.2
best_alpha = optimizeTES(ts, alpha)
print('最优参数：', best_alpha)

mape = Triple_exponential_smoothing(ts, best_alpha)
print('MAPE={:.4%}'.format(mape))



# DataFrame.ewm #指数加权移动平均，即指数平滑
    # (com=None, span=None, halflife=None, alpha=None, 
    #           min_periods=0, adjust=True, ignore_na=False, axis=0)
    # com : float, optional
    # Specify decay in terms of center of mass, 𝛼=1/(1+𝑐𝑜𝑚), for 𝑐𝑜𝑚≥0
    # α  = 1 / (1 + com), for com ≥ 0
    # span : float, optional
    # Specify decay in terms of span, 𝛼=2/(𝑠𝑝𝑎𝑛+1), for 𝑠𝑝𝑎𝑛≥1
    # halflife : float, optional
    # Specify decay in terms of half-life, 𝛼=1−𝑒𝑥𝑝(𝑙𝑜𝑔(0.5)/ℎ𝑎𝑙𝑓𝑙𝑖𝑓𝑒), for ℎ𝑎𝑙𝑓𝑙𝑖𝑓𝑒>0
    # alpha : float, optional。平滑因子0<𝛼≤1
    # min_periods : int, default 0
    # Minimum number of observations in window required to have a value (otherwise result is NA).
    # adjust : bool, default True
    #   当为False时，假定历史数据是无限的，简化计算公式
    #   当为True时，假定历史数据是有限的，计算公式比较复杂
    # ignore_na : bool, default False。计算权重时，是否忽略缺失值。
