#-*- coding: utf-8 -*-

########  本文件实现移动平均，包括
#   Part1 一次移动平均/简单移动平均
#   Part2 二次移动平均
#   Part3 加权移动平均

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from common import displayRegressionMetrics


# 1、读取数据集
filename = '时间序列.xls'
sheet = '餐厅销量'
df = pd.read_excel(filename, sheet)
print(df.columns.tolist())

df.set_index('日期', inplace=True)
ts = df['销量']

# 2、可视化
ts.plot(title='序列')
plt.show()

######################################################################
########  Part1、简单/一次移动平均 Simple Moving Average
######################################################################
# 移动平均：将前N期的平均值（算术平均、加权平均）作为下一期的预测值

# 3、移动平均
N = 3
roll_mean = ts.rolling(window=N).mean()
pred = roll_mean.shift(1)
pred.dropna(inplace=True)

# 4、评估模型
y = ts[pred.index]
mape = np.abs( (y - pred )/y ).mean()
print('MAPE={:.2%}'.format(mape) )

displayRegressionMetrics(y, pred)

# 5、寻找最佳期数N：使得MAPE最小的期数
def sma(ts, N):
    roll_mean = ts.rolling(window=N).mean()
    pred = roll_mean.shift(1)
    pred.dropna(inplace=True)
    
    y = ts[pred.index]
    mape = np.abs( (y-pred) /y ).mean()

    return mape

mapes = []
start, end = 2, 10
for n in range(start, end):
    mape = sma(ts, n)
    mapes.append(mape)

bestN = np.argmin(mapes) + start
print('最佳期数N=', bestN)
print('mape=', mapes[bestN-start])


######################################################################
########  Part2、二次移动平均 Double Moving Average
######################################################################

def double_moving_average(ts, n, isPlot=True):
    m1 = ts.rolling(n).mean()
    m2 = m1.rolling(n).mean()

    a = 2*m1 - m2
    b = 2*(m1 - m2)/(n-1)

    yhat = a + b
    ret = yhat[-1]

    y_pred = yhat.shift(1)
    y_pred.dropna(inplace=True)
    y_true = ts[yhat.index]

    # y_true = ts[2*n-1:]   #去掉前面2*n个数
    # y_pred = y_[2*n-2:-1]

    # 计算误差率
    mape = (np.abs(y_pred - y_true)/y_true).mean()

    if isPlot:
        # 可视化图形
        plt.plot(range(len(y_pred)), y_pred, 'b', label='移动平均')
        plt.plot(range(len(y_true)), y_true, 'g', label='实际值')
        
        plt.legend(loc='upper right')
        title = '二次移动平均(N={})'.format(n)
        plt.title(title)
        plt.show()

    return ret, mape

n = 4
ret,mape = double_moving_average(ts, n)
print('MAPE={:.4%}'.format(mape))

# 最佳期数（略）


######################################################################
########  Part3、加权移动平均 Weighted Moving Average
######################################################################
# y(t) = w1*y(t-1) + w2*y(t-2) + ... + wn*y(t-n)
# weights = [w1, w2, ..., wn]

# 3.训练模型
def weighted_average(ts, weights, isPlot=True):
    """
    计算加权移动平均，返回最近一期的预测值
    """
    wma = lambda x: (x*weights).sum()
    n = len(weights)    #自动取期数
    
    wma_mean = ts.rolling(n).apply(wma)
    ret = wma_mean[-1]          #最后一期预测值
    # y_pred = wma_mean[n-1:-1]
    # y_true = ts[n:]
    y_pred = wma_mean.shift(1)
    y_pred.dropna(inplace=True)
    y_true = ts[y_pred.index]

    # 计算误差率
    mape = (np.abs(y_pred - y_true)/y_true).mean()

    if isPlot:
        # 可视化图形
        plt.plot(range(len(y_pred)), y_pred, 'b', label='移动平均')
        plt.plot(range(len(y_true)), y_true, 'g', label='实际值')
        
        plt.legend(loc='upper right')
        title = '加权移动平均({})'.format(weights)
        plt.title(title)
        plt.show()

    return ret, mape

weights = [0.1, 0.3, 0.6]
ret,mape = weighted_average(ts, weights)
print('MAPE={:.4%}'.format(mape))

# 5.模型优化：最优权重参数
import scipy.optimize as spo
import sklearn.metrics as mts

def optimizeWeight(ts, weights ):
    """
    加权移动平均算法，寻找最优权重
    """
    # 匿名函数
    wma = lambda x: (x*weights).sum()

    # 定义误差函数（参数:数组，其它参数:元组）
    def error(weights, *var):
        ts = var[0]
        n = len(weights)
        # print(weights)
        wma_mean = ts.rolling(n).apply(wma)
        pred = wma_mean.shift(1)
        pred.dropna(inplace=True)
        mse = mts.mean_squared_error(ts[pred.index], pred)
        # print(weights, mse)
        return mse

    # 约束条件
    # type='eq',表示fun值=0；type='ineq',表示fun值为非负数
    cons = ({'type':'eq', 'fun': lambda wts: wts.sum()-1})
    bnds = [(0,1)]*len(weights)   #单个变量范围

    # 优化
    optResult = spo.minimize(error, weights, args=(ts,),
                method='trust-constr',
                bounds=bnds, constraints=cons, 
                # options={'maxiter':1000,'disp':True}
                )
    assert(optResult['success'])
    print(optResult)
    ret = optResult['x']
    return ret   #只返回最优参数

weights = [0.1, 0.2, 0.4, 0.3]
bestWts = optimizeWeight(ts, weights)
print('最优参数：', bestWts)

ret, mape = weighted_average(ts, bestWts)
print('MAPE={:.4%}'.format(mape))


# DataFrame.rolling
    # (window, min_periods=None, center=False, 
    # win_type=None, on=None, axis=0, closed=None)
    # 窗口函数，截取N列数据窗口，以便进行后续的操作。
    # window : int, or offset,窗口偏移的位置
    # min_periods : int, default None
    # Minimum number of observations in window required to have a value (otherwise result is NA). For a window that is specified by an offset, min_periods will default to 1. Otherwise, min_periods will default to the size of the window.
    # center : bool, default False
    # Set the labels at the center of the window.
    # win_type : str, default None
    # Provide a window type. If None, all points are evenly weighted. See the notes below for further information.
    # on : str, optional
    # For a DataFrame, column on which to calculate the rolling window, rather than the index
    # axis : int or str, default 0
    # closed : str, default None
    # Make the interval closed on the ‘right’, ‘left’, ‘both’ or ‘neither’ endpoints. For offset-based windows, it defaults to ‘right’. For fixed windows, defaults to ‘both’. Remaining cases not implemented for fixed windows.
    # New in version 0.20.0.
