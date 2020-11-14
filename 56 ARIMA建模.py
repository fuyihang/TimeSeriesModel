#-*- coding: utf-8 -*-

########  本文件介绍ARIMA相关的模型，包括
#   Part1 序列平稳性检验
#   Part2 白噪声检验
#   Part3 画ACF图和PACF图
#   Part4 AR自回归模型
#   Part5 MA移动平均模型 
#   Part6 ARMA移动平均模型 
#   Part7 差分运算
#   Part8 ARIMA模型
#   Part9 ARIMA模型(季节性)
#   Part10 SARIMA季节模型

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


######################################################################
########  Part1、序列平稳性检验 
######################################################################

# 平稳性检验
#迪基-福勒检验(Augmented Dickey-Fuller test)检验
from statsmodels.tsa.stattools import adfuller

def test_stationarity(ts):
    '''\
        采用ADF检验法，进行序列的平稳性检验
        返回True或False
    '''
    dftest = adfuller(ts)
    p = dftest[1]
    if p < 0.05:
        print("序列是平稳的")
        return True
    else:
        print('序列不是平稳！')
        return False

    # dftest返回值依次保存的是：
        # 'Test Statistic'
        # 'p-value'
        # '#Lags Used'
        # 'Number of Observations Used'
        # 第5个元组是一个字典，保存不同置信度对应的统计值
        # for key,value in dftest[4].items():
        #     srOutput['Critical Value (%s)'%key] = value

    # # 也可采用KPSS Test
    # from statsmodels.tsa.stattools import kpss
    # ret = kpss(ts)
    # p = ret[1]


# 1、检验序列1的平稳性

filename = '时间序列.xls'
sheet = 'ts1'       #有ts1~ts4
df = pd.read_excel(filename, sheet)

ts = df['序列值']
ts[:100].plot()
ret = test_stationarity(ts)

# filename = '../dataset/时间序列.xls'
# for i in range(1, 5):
#     sheet = 'ts{}'.format(i)
#     df = pd.read_excel(filename, sheet)

#     ts = df['序列值']
#     # ts[:100].plot()
#     print(sheet, end=' ')
#     ret = test_stationarity(ts)


# 2、检验沃尔玛销量数据的平稳性

filename = '时间序列.xls'
sheet = 'Walmartdata'
df = pd.read_excel(filename, sheet)

ts = df['销量']
ts[:100].plot()
ret = test_stationarity(ts)

######################################################################
########  Part2、白噪声检验 
######################################################################
# 白噪声是一种特殊的平稳序列，纯随机，均值恒定为0，且无自相关性

# 白噪声检验/纯随机检验（杨博克斯检验）
from statsmodels.stats.diagnostic import acorr_ljungbox

def test_stochastic(ts, lags=None):   #lags可自定义
    # 1)计算平均值和标准差
    avg = np.mean(ts)
    std = np.std(ts)
    print('mean={:.2f},std={:.2f}'.format(avg, std))

    # 2)用方法检验
    lbtest= acorr_ljungbox(ts, lags=lags)
    p_value = lbtest[1] #只取pvalue
    
    ret = True
    for p in p_value:
        if p < 0.05:
            print("序列不是白噪声！")
            ret = False
            break
    else:
        print("序列是白噪声。")

    return ret

# 1、检验序列是否是白噪声
num = 1000
e = np.random.standard_normal(num) #构造白噪声
ts = pd.Series(e)
print('mean=', np.mean(ts))
print('std=', np.std(ts))

ts[:100].plot()
ret = test_stochastic(ts)


# # 2、白噪声检验
# filename = '时间序列.xls'
# for i in range(1, 5):
#     sheet = 'ts{}'.format(i)
#     df = pd.read_excel(filename, sheet)

#     ts = df['序列值']
#     # ts[:100].plot()
#     print(sheet, end=' ')
#     ret = test_stochastic(ts)


######################################################################
########  Part3、画ACF图和PACF图
######################################################################

# 检验
# from statsmodels.tsa.stattools import acf, pacf     #计算函数
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf   #画图函数

# 请分别画出ts1~ts4的ACF和PACF图
filename = '时间序列.xls'
for i in range(1, 5):
    sheet = 'ts{}'.format(i)
    df = pd.read_excel(filename, sheet)

    ts = df['序列值']

    fig = plot_acf(ts)
    fig = plot_pacf(ts, lags=40)


######################################################################
########  Part4、AR自回归模型 
######################################################################

# 一、对ts1进行建模

# 1、读取数据
filename = '时间序列.xls'
sheet = 'ts1'
df = pd.read_excel(filename, sheet)

ts = df['序列值']
ts[:100].plot()

# 2、检验平稳性
ret = test_stationarity(ts)

# 3、模型识别与定阶

fig = plot_acf(ts)
fig = plot_pacf(ts)
# ACF拖尾，PACF一阶截尾，为AR(1)模型,p=1
p, q = 1, 0

# 4、建立AR(p)模型
# from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA

mdl = ARMA(ts, order=(p,q))
results = mdl.fit()

# 查看模型的系数
print('AR系数', results.params)

# 相当于 yt = -0.5 * yt-1 + e 

# 打印详细信息
print(results.summary())

# 5、评估模型

# 1)AIC/BIC指标
print('AIC=', results.aic)
print('BIC=', results.bic)
print('HQIC=', results.hqic)

# 2)评估回归指标
ts_pred = results.fittedvalues       #返回预测结果
# ts_pred = results.predict()        #同样结果

# 注意：pred.index与ts.index是对应的，但要少了前p个数
ts_tmp = ts[ts_pred.index]

displayRegressionMetrics(ts_tmp, ts_pred)

# 3）可视化,看前N个容易看清
plt.figure()
N = 100
plt.plot(ts_tmp[:N], c='g')
plt.plot(ts_pred[:N], c='b')
plt.show()

# # 或者,这个不好看
# results.plot_predict()
# plt.show()

# 4)检查残差是否是白噪声
# 残差result.resid = ts_tmp - ts_pred
resid = results.resid
ret = test_stochastic(resid)


# 6、应用模型
# print(ts.index)

# 1）预测当前时期的数据
# pred = results.predict()   #表示所有时期
pred = results.predict(start=1990,end=2000)  #返回series
print(pred)

# 2）预测未来N=5期的数据
pred = results.forecast(steps=5)[0] #返回数组
print(pred)

# 3)保存模型
filename = 'out.pkl'
results.save(filename)

# 4)使用时，装载模型
from statsmodels.tsa.arima_model import ARIMAResults
filename = 'out.pkl'
results = ARIMAResults.load(filename)

print(results.bic)

# 二、请对ts2进行建模
# ts2:AR(2)


######################################################################
########  Part5、MA移动平均模型 
######################################################################

# 一、对ts3进行建模

# 1、读取数据
filename = '时间序列.xls'
sheet = 'ts3'
df = pd.read_excel(filename, sheet)

ts = df['序列值']
ts[:100].plot()

# 2、检验平稳性
ret = test_stationarity(ts)

# 3、模型识别与定阶

fig = plot_acf(ts)
fig = plot_pacf(ts)
# ACF三阶截尾，PACF拖尾，为MA(3)模型,q=3
p, q = 0, 3

# 4、建立MA(q)模型
from statsmodels.tsa.arima_model import ARMA

mdl = ARMA(ts, order=(p,q))
results = mdl.fit() #

# 查看模型的系数
print('MA系数', results.maparams)

# 查看详细信息
print(results.summary())

# 5、评估模型（同前，略）
print('BIC=', results.bic)
ret = test_stochastic(results.resid)

# 6、应用模型（同前，略）

######################################################################
########  Part6、ARMA移动平均模型 
######################################################################

# 一、对ts4进行建模

# 1、读取数据
filename = '时间序列.xls'
sheet = 'ts4'
df = pd.read_excel(filename, sheet)

ts = df['序列值']
ts[:100].plot()

# 2、检验平稳性
ret = test_stationarity(ts)

# 3.1 模型识别：ACF/PACF图形

fig = plot_acf(ts)
fig = plot_pacf(ts)

# ACF拖尾，PACF也拖尾，为ARMA模型

# 3.2 模型定阶：AIC/BIC信息准则定阶
import statsmodels.api as sm

results = sm.tsa.arma_order_select_ic(ts, ic=['bic'],
            max_ar=6, max_ma=6)
print('BIC_order:', results['bic_min_order'])

p, q = results['bic_min_order']

# 也可以用其它信息准则
# results = sm.tsa.arma_order_select_ic(ts, ic=['aic','bic','hqic'],
#             trend='nc',max_ar=4, max_ma=4)
# print('BIC', results['bic_min_order'])
# print('HQIC', results['hqic_min_order'])

# 5、建立ARMA(p,q)模型
from statsmodels.tsa.arima_model import ARMA

mdl = ARMA(ts, order=(p,q))
results = mdl.fit() #

# 查看模型的系数
print('ARMA系数', results.params)
print('AR系数', results.arparams)
print('MA系数', results.maparams)

# 6、评估模型（同前，略）
ret = test_stochastic(results.resid)

# 7、应用模型（同前，略）


######################################################################
########  Part7、差分运算 
######################################################################

# 一、对"上海证券交易数据"进行建模

# 1、读取数据
filename = '时间序列.xls'
sheet = '上海证券交易数据'
df = pd.read_excel(filename, sheet)

ts = df['收盘价']
ts.index = pd.to_datetime(df['日期'],format='%Y%m%d')

ts.plot()

# 2、检验平稳性
# 1）检验
ret = test_stationarity(ts)

# 2）看图
fig = plot_acf(ts)
fig = plot_pacf(ts)

# 都发现不平稳

# 3、作差分运算

# 尝试一：作d=1阶差分
diff = ts.diff(1)   #1表示1步差分，diff一阶差分
diff.dropna(inplace=True)

ret = test_stationarity(diff)

# 4、模型识别与定阶
fig = plot_acf(diff, lags=40)
fig = plot_pacf(diff, lags=40)

# 5、建立模型
from statsmodels.tsa.arima_model import ARMA
mdl = ARMA(diff, order=(11,11))
results = mdl.fit()

print('系列列表:\n', results.params)
print('AR系数：\n', results.arparams)
print('MA系数：\n', results.maparams)

# 6、评估
# 1）mape 评估指标
# 2）残差的白噪声检验
ret = test_stochastic(results.resid)

# 7、预测
# 预测过去
pred = results.predict(start='20090728', end='20090731',dynamic=True)
# pred = results.predict()  #默认预测历史数据
print(pred)

# 预测未来的N=3个值，#函数返回值中批一个数组为预测值
pred_future = results.forecast(steps=3)[0]
print(pred_future)

# 8、还原
# 因为前面diff = ts.diff(1) 相当于diff[t] = ts[t] - ts[t-1]
# 即 ts[t-1] = ts[t] - diff[t]
# 再将ts[t-1]上升一位，即成了ts[t]
# 一行代码：ts = (ts - diff).shift(-1)搞定
pred = results.predict()    #对diff序列的预测

# 现在，pred是diff的预测值, 要将预测值还原为ts值
ts_old = (ts - pred).shift(-1)
ts_old.dropna(inplace=True)

plt.plot(ts[:-1], label='原始值')
plt.plot(ts_old, label='预测值')
plt.legend()
plt.show()


# 还原未来的值
N = len(pred_future) + 1
ts_old = [0]*N
ts_old[0] = ts.iloc[-1]
for i in range(len(pred_future)):
    ts_old[i+1] = pred_future[i] + ts_old[i]  # ts[t] = diff[t] + ts[t-1]
print(ts_old)

ts_old = ts_old[1:] #去掉前N个数据

######################################################################
########  Part8、ARIMA模型 
######################################################################
# 上述操作也可以优化一下，让模型自己处理差分

# 1、读取数据
filename = '时间序列.xls'
sheet = '上海证券交易数据'
df = pd.read_excel(filename, sheet)

ts = df['收盘价']
ts.index = pd.to_datetime(df['日期'],format='%Y%m%d')

ts.plot()

# 2、检验平稳性
# 1）检验
ret = test_stationarity(ts)

# 3、作差分运算(略)

# 4、模型识别与定阶
fig = plot_acf(diff, lags=40)
fig = plot_pacf(diff, lags=40)

# 5、建立模型
from statsmodels.tsa.arima_model import ARIMA
mdl = ARIMA(ts, order=(11,1,11))    #注意传入的是原始的ts
results = mdl.fit()

print('AR系数：\n', results.arparams)
print('MA系数：\n', results.maparams)

# 6、评估
print('BIC=', results.bic)
ret = test_stochastic(results.resid)

ts_pred = results.predict()

# 居然返回的是差分后的预测值？？？？
ts_old = (ts - ts_pred).shift(-1)
ts_old.dropna(inplace=True)

ts_tmp = ts[ts_old.index]

######################################################################
########  Part9、SARIMA--带季节的ARIMA模型 
######################################################################

# 1、读取数据
filename = '时间序列.xls'
sheet = 'AirPassengers'
df = pd.read_excel(filename, sheet)

df.set_index('日期', inplace=True)
ts = df['乘客数']

ts.plot()

# 2、检验平稳性
# 1）检验
ret = test_stationarity(ts)

# 3、平稳化处理：差分运算
# 1)先将数值变小
ts_log = np.log(ts)
ret = test_stationarity(ts_log)

# 2）作移动平均规则化
T = 12
ts_mean = ts_log.rolling(window=T).mean()
ts_mean.dropna(inplace=True)
ret = test_stationarity(ts_mean)

# # 3.1)作一阶差分运算
# diff = ts_mean.diff(1)
# diff.dropna(inplace=True)
# ret = test_stationarity(diff)

# 3.2)作二阶差分
diff_1 = ts_mean.diff(1)
diff = diff_1.diff(1)
diff.dropna(inplace=True)
ret = test_stationarity(diff)

# # 第二种平稳化处理
# # 2)观察图形存在季节周期
# T = 12
# ts_k12 = ts_log.diff(T)
# ts_k12.dropna(inplace=True)
# ret = test_stationarity(ts_k12)

# # 3)作差分运算
# diff = ts_k12.diff(1)
# diff.dropna(inplace=True)
# ret = test_stationarity(diff)
# diff.plot()

# 4、模型识别与定阶
fig = plot_acf(diff, lags=40)
fig = plot_pacf(diff, lags=40)

p,q = 12,12
from statsmodels.tsa.arima_model import ARMA

mdl = ARMA(diff, order=(p,q))
results = mdl.fit()
print(results.params)

# 5、评估模型
print('BIC=', results.bic)
ret = test_stochastic(results.resid)
# mape =

# 6、预测值还原
pred = results.predict()

# 二阶差分还原
diff_1_old = (diff_1 - pred).shift(-1)  #还原在差分一阶

# 一阶差分还原
ts_mean_old = (ts_mean - diff_1_old).shift(-1)

# 移动平均还原
# ts_mean_old.dropna(inplace=True)
rol_sum = ts_log.rolling(window=T-1).sum()
ts_log_old = ts_mean_old*12 - rol_sum.shift(1)

# 对数还原
ts_old = np.exp(ts_log_old)
ts_old.dropna(inplace=True)

# 可视化对比
plt.plot(ts_old,c='b')
plt.plot(ts[ts_old.index], c='g')
plt.show()

######################################################################
########  Part10、SARIMA 季节模型 
######################################################################
# SARIMA(p, d, q)(P,D,Q,S)

# 1、读取数据
filename = '时间序列.xls'
sheet = 'AirlineMiles'
df = pd.read_excel(filename, sheet)

df.set_index('日期', inplace=True)
ts = df['里程(万)']

ts.plot()

# 2、检验平稳性
# 1）检验
ret = test_stationarity(ts)

# 2）差分运算
diffd = ts.diff(1)
diffd.dropna(inplace=True)
ret = test_stationarity(diffd)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig = plot_acf(diffd)
fig = plot_pacf(diffd)

# 对于非季节项，做了一阶差分，即d=1，
# 再看ACF和PACF，初步可定p=5或6,q=6

# 3)K步差分
diffk = diffd.diff(12)
diffk.dropna(inplace=True)
ret = test_stationarity(diffk)

fig = plot_acf(diffk)
fig = plot_pacf(diffk)

# 对于季节项，做了一阶季节差分，即D=1，
# 再看ACF和PACF，初步可定P=1,Q=1或3

# 3、建立季节模型
from statsmodels.tsa.statespace.sarimax import SARIMAX

p,d,q = 6,1,6
P,D,Q,T = 1,1,3,12
mdl = SARIMAX(ts, 
            order=(p, d, q), 
            seasonal_order=(P, D, Q, T),
            enforce_stationarity=False,
            enforce_invertibility=False)
results = mdl.fit()

# 4、评估模型
ret = test_stochastic(results.resid)
pred = results.predict()[1:]    #第一个值为0，去除
print(pred)

mape = np.abs(results.resid/ts[results.resid.index]).mean()
print('MAPE={:.2%}'.format(mape) )

# 5、优化，手工遍历最优的(p,d,q)(P,D,Q,S)
import itertools
p=d=q=range(2)
T = 12
pdq = list(itertools.product(p,d,q))
seasonal_pdq = [(x[0], x[1], x[2], T) for x in pdq]

min_bic = np.inf
best_pdq = (0,0,0)
best_pdqs = (0,0,0,T)
for param in pdq:
    print(param)
    for param_seasonal in seasonal_pdq:
        try:
            mdl = SARIMAX(ts, 
                        order = param,
                        seasonal_order = param_seasonal,
                        enforce_stationarity = False,
                        enforce_invertibility = False)
            results = mdl.fit()
            if results.bic < min_bic:
                min_bic = results.bic
                best_pdq = param
                best_pdqs = param_seasonal
        except:
            continue
print('最小BIC=', min_bic)
print(best_pdq, best_pdqs)

(p, d, q) = best_pdq
(P, D, Q, T) = best_pdqs
mdl = SARIMAX(ts, 
            order=(p, d, q), 
            seasonal_order=(P, D, Q, T),
            enforce_stationarity=False,
            enforce_invertibility=False)
results = mdl.fit()

pred = results.fittedvalues[1:]
ret = test_stochastic(results.resid[1:])
# results.plot_diagnostics()


######################################################################
########  Part、实战 
######################################################################

