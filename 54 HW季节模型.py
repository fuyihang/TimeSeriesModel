#-*- coding: utf-8 -*-

########  本文件实现Holt-Winters季节预测模型，包括
#   Part1 季节因素分解
#   Part2 Holt-Winters additive method
#   Part3 Holt-Winters Multiplicative method
#   Part4 Holt-Winters Exponential method

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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


######################################################################
########  Part1、季节因素分解
######################################################################
# 即将时序分解成长期趋势、季节趋势和随机成分
# 加法additive:Y[t] = T[t] + S[t] + e[t]
# 乘法multiplicative:Y[t] = T[t] * S[t] * e[t]

# 季节分解函数
from statsmodels.tsa.seasonal import seasonal_decompose
# seasonal_decompose(x, 
            # model='additive', 'multiplicative'
            # filt=None,
            # period=None, 
            # two_sided=True, 
            # extrapolate_trend=0   为'freq'时有助于处理序列首部趋势和残差中的空值
            # )

# 序列分解模型
def showCompose(ts, model='additive', period=None):
    decomp = seasonal_decompose(ts, period=period, model=model)

    plt.figure()
    plt.subplot(411)
    plt.plot(ts, label='原始数据')

    plt.subplot(412)
    plt.plot(decomp.trend, label='趋势')

    plt.subplot(413)
    plt.plot(decomp.seasonal, label='季节')
    print(decomp.seasonal[:period])

    plt.subplot(414)
    plt.plot(decomp.resid, label='残差')

    plt.legend()
    plt.show()


# 1、航空里程
filename = '时间序列.xls'
sheet = 'AirlineMiles'
df = pd.read_excel(filename, sheet)

df.set_index('日期', inplace=True)
ts1 = df['里程(万)']
ts1.plot()

showCompose(ts1, 'add',period=12)
showCompose(ts1, 'mul',period=12)

# 2、航空人数
filename = '时间序列.xls'
sheet = 'AirPassengers'
df = pd.read_excel(filename, sheet)

df.set_index('日期', inplace=True)
ts2 = df['乘客数']
ts2.plot()

showCompose(ts2, 'add',period=12)
showCompose(ts2, 'mul',period=12)

# #消除了trend 和seasonal之后，
# 只对residual部分作为想要的时序数据进行ARIMA建模

######################################################################
########  Part2、 Holt-Winters additive method
######################################################################

from statsmodels.tsa.holtwinters import ExponentialSmoothing
# ExponentialSmoothing(endog, trend=None, damped=False, 
            # seasonal=None, seasonal_periods=None, 
            # dates=None, freq=None, missing='none')
    # HW加法：（trend='add', seasonal='add', damped=False）
    # HW乘法：（trend='add', seasonal='mul', damped=False）
    # HW指数：（trend='mul', seasonal='mul', damped=False）

# ExponentialSmoothing.fit(
    # smoothing_level=None,     指定alpha
    # smoothing_slope=None,     指定beta
    # smoothing_seasonal=None,  指定gamma
    # damping_slope=None,       指定phi
    # optimized=True, 
    # use_boxcox=False,     先执行Box-Cox变换
    # remove_bias=False, 
    # use_basinhopping=False, 
    # start_params=None, 
    # initial_level=None, initial_slope=None, 
    # use_brute=True)

# 1、读取数据集
filename = '时间序列.xls'
sheet = 'AirlineMiles'
df = pd.read_excel(filename, sheet)

df.set_index('日期', inplace=True)
ts = df['里程(万)']

# 2、可视化
ts.plot(title=sheet)
plt.show()

# 3、训练模型
T = 12
mdl = ExponentialSmoothing(ts, trend='add', 
                            seasonal='add',
                            seasonal_periods=T
                            )
results = mdl.fit()
print(results.params)   #可取出最优平滑参数（略）
# print(results.summary())

# 4.评估模型（略）
y_pred = results.fittedvalues
y_true = ts
displayRegressionMetrics(y_true, y_pred)


# 6.应用模型（略）
# 1）预测历史
pred = results.predict(start='2012-01-01', end='2012-06-01')
print(pred)

# 2）预测未来
pred = results.forecast(steps=5)
print(pred)

# 3）保存模型
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
########  Part3、 Holt-Winters Multiplicative method
######################################################################

# 1、读取数据集
filename = '时间序列.xls'
sheet = 'AirPassengers'
df = pd.read_excel(filename, sheet)

df.set_index('日期', inplace=True)
ts = df['乘客数']

# 2、可视化
ts.plot(title=sheet)
plt.show()

# 3、训练模型
T = 12
mdl = ExponentialSmoothing(ts, trend='add', 
                            seasonal='mul',
                            seasonal_periods=T
                            )
results = mdl.fit()
print(results.params)   #可取出最优平滑参数（略）
# print(results.summary())

# 4.评估模型（略）
# 6.应用模型（略）

######################################################################
########  Part4、 Holt-Winters Exponential method
######################################################################


# 1、读取数据集
filename = '时间序列.xls'
sheet = 'Walmartdata'
df = pd.read_excel(filename, sheet)

# 2、预处理
col = '日期'
df[col] = df[col].astype('datetime64[ns]')
df.set_index(col, inplace=True)

ts = df['销量'].to_period('Q')

# 2、可视化
ts.plot(title=sheet)
plt.show()

# 3、训练模型
T = 4
mdl = ExponentialSmoothing(ts, trend='mul', 
                            seasonal='mul',
                            seasonal_periods=T
                            )
results = mdl.fit()
print(results.params)   #可取出最优平滑参数（略）
# print(results.summary())

# 4.评估模型（略）
# 6.应用模型（略）


# 重要参数
    # trend,{"add", "mul", "additive", "multiplicative", None}
    # damped : bool, optional Should the trend component be damped
    # seasonal : {"add", "mul", "additive", "multiplicative", None},
    # seasonal_periods : int, 季节周期（季节为4，周为7）

