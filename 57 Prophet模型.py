#-*- coding: utf-8 -*-

########  本文件实现Prophet模型，包括
#   预处理、网络结构、超参优化、滚动预测

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pip install  fbprophet
# 或者conda install -c conda-forge fbprophet


######################################################################
########  Part1、Prophet模型 
######################################################################

# 1、读取数据
filename = '时间序列.xls'
sheet = '上海证券交易数据'
df = pd.read_excel(filename, sheet)
df = df[['日期', '收盘价']]

# Prophet必须要求列名为ds和y
df.rename(columns={'日期':'ds', '收盘价':'y'}, inplace=True)

ts = df['y']
ts.plot()

# 2、训练模型
from fbprophet import Prophet

# 默认使用线性模型，当预测增长时，可以指定growth='logistic'
mdl = Prophet(growth = 'linear')
mdl.fit(df)             #help(Prophet.fit)

# 3、评估模型
pred = mdl.predict(df['y'])
mape = np.abs( (ts - pred)/ts).mean()


# 4、预测值
future = mdl.make_future_dataframe(periods = 5, freq='D')
pred = mdl.predict(future)
print(pred.columns.tolist())

print(pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# 画预测图形
mdl.plot(pred)

# 画预测成分图形
mdl.plot_components(pred)
# pred['yhat']  = pred['trend'] +pred['weekly'] + pred['yearly'] + pred['holidays']


# 二、节假日(指定)
playoffs = pd.DataFrame({
  'holiday': '促销日',
  'ds': pd.to_datetime(['2004-01-13', '2005-01-03', '2006-01-16',
                        '2006-01-24', '2006-02-07', '2007-01-08',
                        '2008-01-12', '2009-01-12']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': '国庆节',
  'ds': pd.to_datetime(['2006-02-07', '2009-02-02']),
  'lower_window': 0,
  'upper_window': 1,
})

holidays = pd.concat((playoffs, superbowls))

mdl = Prophet( holidays = holidays)
mdl.fit(df)
future = mdl.make_future_dataframe(periods = 15)
pred = mdl.predict(future)
print(pred.columns.tolist())

# 看一下假期的最后5行数据
pred[(pred['playoff'] + pred['superbowl']).abs() > 0][['ds', 'playoff', 'superbowl']][-5:]


# 三、季节周期
mdl = Prophet()
mdl.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.1)
mdl.add_seasonality(name='monthly', period=30.5, fourier_order=5)
mdl.fit(df)
# period表示周期数，傅立叶阶数fourier_order，prior_scale表示设置先验规模,避免过拟合


# 四、季节性(指数增长)，指定参数seasonality_mode='additive'
mdl = Prophet(seasonality_mode='multiplicative')


# 超参优化

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler

# def ts_evaluation(df, param, horizon=30, period=120, initial=1095, exp=True):
#     '''
#     利用交叉验证评估效果   
#     '''
#     #param = {'holidays':holidays,'growth':'linear','seasonality_prior_scale':50,'holidays_prior_scale':20}
#     mdl = Prophet(**param)
#     mdl.fit(df)
#     forecasts = mdl.predict(df[['ds']])
#     forecasts['y'] = df['y']
#     df_cv = cross_validation(mdl, 
#             horizon='{} days'.format(horizon), 
#             period='{} days'.format(period), 
#             initial='{} days'.format(initial))
#     if exp:
#         df_cv['yhat']=np.exp(df_cv['yhat'])
#         df_cv['y']=np.exp(df_cv['y'])
#     mape = np.mean(np.abs((df_cv['y'] - df_cv['yhat']) / df_cv['y']))
#     rmse = np.sqrt(np.mean((df_cv['y'] - df_cv['yhat'])**2))
#     scores = {'mape':mape,'rmse':rmse}
#     return scores

# def ts_grid_search(df, holidays, param_grid=None, cv_param=None, 
#           RandomizedSearch=True, random_state=None):
#     '''网格搜索
#     时间序列需要特殊的交叉验证
#     df:
#     holidays: 需要事先调好  
#     '''

#     df=df.copy()
#     if param_grid is None:
#         param_grid={'growth':['linear']
#         ,'seasonality_prior_scale':np.round(np.logspace(0,2.2,10))
#         ,'holidays_prior_scale':np.round(np.logspace(0,2.2,10))
#         ,'changepoint_prior_scale':[0.005,0.01,0.02,0.03,0.05,0.008,0.10,0.13,0.16,0.2]
#         }

#     if RandomizedSearch:
#         param_list=list(ParameterSampler(param_grid, n_iter=10, random_state=random_state))
#     else:
#         param_list=list(ParameterGrid(param_grid))

#     if cv_param is None:
#         cv_param={'horizon':30, 'period':120, 'initial':1095}

#     scores=[]
#     for i,param in enumerate(param_list):
#         print('{}/{}:'.format(i, len(param_list)), param)
#         param.update({'holidays':holidays})
#         scores_tmp = ts_evaluation(df, param, exp=True, **cv_param)        
#         param.pop('holidays')
#         tmp = param.copy()
#         tmp.update({'mape':scores_tmp['mape'], 'rmse':scores_tmp['rmse']})
#         scores.append(tmp)                       
#         print('mape : {:.5f}%'.format(100*scores_tmp['mape']))


#     scores=pd.DataFrame(scores)

#     best_param_=scores.loc[scores['mape'].argmin(),:].to_dict()
#     best_scores_=best_param_['mape']
#     best_param_.pop('mape')
#     best_param_.pop('rmse')


#     return best_param_,best_scores_,scores

