#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 特征选择：利用相关性
import statsmodels.formula.api as smf
import statsmodels.stats.anova as smsa
def featureSelection(df, intCols, catCols, target, threshold=0.05):
    '''\
        实现特征选择，利用协方差分析
    '''
    # 仅做主效应检验
    formula = target + ' ~ ' + \
                '+'.join(intCols) + '+' + \
                '+'.join(catCols)                

    module = smf.ols(formula, df).fit()
    dfRet = smsa.anova_lm(module)

    # 取显著因子项
    cond = dfRet['PR(>F)'] < threshold
    cols = dfRet[cond].index.tolist()
    # print(cols)

    # 筛选变量
    for col in intCols:
        if col not in cols:
            intCols.remove(col)
    for col in catCols:
        if col not in cols:
            catCols.remove(col)
    return intCols, catCols

def intFeatureSelection(df, intCols, target, threshold=0.3):
    '''\
        实现数值型变量的特征选择，利用相关系数矩阵
    '''
    dfcorr = df[intCols+[target]].corr(method='spearman')

    cond = np.abs(dfcorr[target]) > threshold
    cols = dfcorr[cond].index.tolist()
    cols.remove(target)
    return cols

# 显示回归评估指标
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

    # print('回归模型评估指标：\n', mts)
    
    return mts


# 画学习曲线
from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator, X, y, cv=None, scoring = None):
    train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv = cv,scoring=scoring, 
            train_sizes=np.linspace(0.1, 1.0, 50))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("学习网线")
    plt.xlabel('训练样本数量')
    plt.ylabel(scoring)
    plt.grid()

    skipNum = 3
    plt.fill_between(train_sizes[skipNum:], 
                train_scores_mean[skipNum:] - train_scores_std[skipNum:],
                train_scores_mean[skipNum:] + train_scores_std[skipNum:], 
                alpha=0.1, color="b")
    plt.fill_between(train_sizes[skipNum:], 
                test_scores_mean[skipNum:] - test_scores_std[skipNum:],
                test_scores_mean[skipNum:] + test_scores_std[skipNum:], 
                alpha=0.1, color="g")

    plt.plot(train_sizes[skipNum:], train_scores_mean[skipNum:], 
            'o-', color="b", label="训练集")
    plt.plot(train_sizes[skipNum:], test_scores_mean[skipNum:], 
            'o-', color="r", label="测试集")

    plt.legend(loc="best")
    plt.show()


# 自定义类：实现特征选择、类别变量哑变量化、数值变量标准化
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import statsmodels.formula.api as smf
import statsmodels.stats.anova as sms

class MyFeaturePreprocessing(object):
    def __init__(self,normalize=False):
        super().__init__()
        self.cols = []
        self.catCols = []
        self.intCols = []
        self.target = ''

        self.pthreshold = 0.05
        self.normalize = normalize

        self.encCat = OneHotEncoder(drop='first', sparse=False)
        self.encInt = StandardScaler()

    def fit(self, X, y=None):
        """主要实现如下功能：\n
        1）负责筛选出显著影响的因素
        2）对类别型变量进行哑变量转换
        3）对数值型变量进行标准化处理
        """
        self.cols = []
        self.catCols = []
        self.intCols = []
        self.target = ''

        df = pd.concat([X, y], axis=1)
        # print(df.dtypes)
        self.target = y.name
        cols = X.columns.tolist()

        # 1)自动识别变量类型
        for col in cols:
            if np.issubdtype(df.dtypes[col], np.number):
                self.intCols.append(col)
            else:
                self.catCols.append(col)
        # print(self.intCols,"\n", self.catCols,"\n", self.target)

        # 2)找出显著相关的变量
        formula = '{} ~ {}'.format(
                        self.target,
                        '+'.join(cols))
        module = smf.ols(formula, df).fit()
        dfanova = sms.anova_lm(module)
        # print(dfanova)
        
        cond = dfanova['PR(>F)'] < self.pthreshold
        cols = dfanova[cond].index.tolist()
        # print('显著影响因素：\n',cols)

        for col in self.intCols:
            if col not in cols:
                self.intCols.remove(col)
        for col in self.catCols:
            if col not in cols:
                self.catCols.remove(col)

        # 3）类别变量--哑变量化
        if self.catCols != []:
            # self.encCat = OneHotEncoder(drop='first', sparse=False)
            self.encCat.fit(df[self.catCols], y)

        # 4）数值变量--标准化
        if self.normalize and self.intCols != []:
            # self.encInt = StandardScaler()
            self.encInt.fit(df[self.intCols], y)

        return self

    def transform(self, X, copy=None):
        df = pd.DataFrame(X)

        # 1）类别变量转换
        X_ = self.encCat.transform(df[self.catCols])

        cols = []
        for i in range(len(self.encCat.categories_)):
            cols.extend( self.encCat.categories_[i][1:].tolist() )
        
        dfCats = pd.DataFrame(X_, index=df.index, columns=cols)

        # 2）数值变量转换
        if self.normalize:
            X_ = self.encInt.transform(df[self.intCols])
            dfInts = pd.DataFrame(X_, index=df.index,columns=self.intCols)
        else:
            dfInts = df[self.intCols]
        # 3）合并
        dfRet = pd.concat([dfCats, dfInts], axis=1)
        self.cols = dfRet.columns.tolist()

        return dfRet

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        dfRet = self.transform(X)
        return dfRet
