#-*- coding: utf-8 -*-

########  本文件实现LSTM神经网络预测模型，包括
#   预处理、网络结构、超参优化、滚动预测

# 安装keras步骤
# 1）conda install mingw libpython
# 2）conda install theano
# 3）conda install keras

# conda install tensorflow
# 安装tensorflow
# 需要python3.5版本，创建一个虚拟环境
# conda create -n tensorflow python=3.5
# activate tensorflow
# python -m pip install --upgrade pip
# pip install tensorflow==2.0.0-alpha0
# import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1、读取数据
filename = '时间序列.xls'
sheet = 'AirPassengers'
df = pd.read_excel(filename, sheet, index_col='日期')
target = '乘客数'
print('df.shape=', df.shape)

ts = df[target]
ts.plot()

# 2、预处理数据

# 1）构造新的数据集data:X和Y：
# 假定使用的前shiftN期预测
result = []
shiftN = 3

for i in range(len(ts) - shiftN):
    result.append( ts.iloc[i:i+shiftN+1].values )
print(len(result))

data = np.array(result)

# 2)标准化（否则loss下降会很慢）
from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler()
X = scalerX.fit_transform(data[:, :shiftN])


scalerY = MinMaxScaler()
Y = scalerY.fit_transform(data[:, -1].reshape(-1,1))

# 3)LSTM模型的要求是3维输入(N,W,F)=(samples样本数, timesteps序列长度, features特征数]
trainX = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 3、训练模型
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation

# 假定网络结构为[1, 4, 1],即1个输入层, 隐藏层有4个神经元，输出层一个神经元
mdl = Sequential()
mdl.add( LSTM(4, activation='relu', input_shape=(shiftN,1) ) )
mdl.add( Dense(1, activation='sigmoid') )
mdl.compile(loss='mean_squared_error', optimizer='adam')
mdl.fit( trainX, Y, epochs=100, batch_size=10, verbose=2)

# 一般LSTM模块的层数越多(不超过3层)，学习能力越强
# LSTM的网络结构[1, 50, 10, 1]
# 即1个输入层(shiftN个序列), 
mdl = Sequential()

mdl.add( LSTM(50, activation='relu', input_shape=(shiftN, 1), return_sequences=True))
mdl.add( LSTM(100,return_sequences=False))
mdl.add( Dropout(0.2) )
mdl.add( Dense(output_dim = 1, activation='sigmoid') )
# mdl.add( Activation('linear') )

# 编译、训练模型
mdl.compile(loss='mse', optimizer='rmsprop')
mdl.fit( trainX, Y, epochs=100, batch_size=50, verbose=2)

# 4、评估模型
pred = mdl.predict(trainX)
print('pred.shape=', pred.shape)

# 1)还原预测值
y_pred = scalerY.inverse_transform(pred)

# 2)计算误差
y = ts[shiftN:].values
mape = np.abs( (y - y_pred)/y ).mean()
print('MAPE={:.2%}'.format(mape) )

# 3)画图对比
plt.plot(y, label='原始值')
plt.plot(y_pred, label='预测值')
plt.legend()
plt.show()

# 6、滚动预测未来5期的数据
steps = 5
preds = []

x_tail = ts[-3:].values
x_tail = scalerX.transform(x_tail.reshape(1,3))

# x_tail = trainX[-1]  #取最后三个数
for step in range(steps):
    x = np.reshape(x_tail, (1, shiftN, 1))  #转换成三维
    pred = mdl.predict(x)
    preds.append(pred[0][0])
    x_tail2 = np.concatenate( [x_tail.reshape(3,1), pred], axis=0)
    x_tail = x_tail2[1:,:] # 窗口滑动
print(preds)

# keras.layers.LSTM(
    # units: 正整数，输出空间的维度。
    # activation='tanh': 要使用的激活函数 (详见 activations)。 如果传入 None，则不使用激活函数 (即 线性激活：a(x) = x)。
    # recurrent_activatio='hard_sigmoid'n: 用于循环时间步的激活函数 (详见 activations)。 默认：分段线性近似 sigmoid (hard_sigmoid)。 如果传入 None，则不使用激活函数 (即 线性激活：a(x) = x)。
    # use_bias=True: 布尔值，该层是否使用偏置向量。
    # kernel_initializer='glorot_uniform': kernel 权值矩阵的初始化器， 用于输入的线性转换 (详见 initializers)。
    # recurrent_initializer='orthogonal': recurrent_kernel 权值矩阵 的初始化器，用于循环层状态的线性转换 (详见 initializers)。
    # bias_initializer='zeros':偏置向量的初始化器 (详见initializers).
    # unit_forget_bias=True: 布尔值。 如果为 True，初始化时，将忘记门的偏置加 1。 将其设置为 True 同时还会强制 bias_initializer="zeros"。 这个建议来自 Jozefowicz et al.。
    # kernel_regularizer=None: 运用到 kernel 权值矩阵的正则化函数 (详见 regularizer)。
    # recurrent_regularizer=None: 运用到 recurrent_kernel 权值矩阵的正则化函数 (详见 regularizer)。
    # bias_regularizer=None: 运用到偏置向量的正则化函数 (详见 regularizer)。
    # activity_regularizer=None: 运用到层输出（它的激活值）的正则化函数 (详见 regularizer)。
    # kernel_constraint=None: 运用到 kernel 权值矩阵的约束函数 (详见 constraints)。
    # recurrent_constraint=None: 运用到 recurrent_kernel 权值矩阵的约束函数 (详见 constraints)。
    # bias_constraint=None: 运用到偏置向量的约束函数 (详见 constraints)。
    # dropout=0.0: 在 0 和 1 之间的浮点数。 单元的丢弃比例，用于输入的线性转换。
    # recurrent_dropout=0.0: 在 0 和 1 之间的浮点数。 单元的丢弃比例，用于循环层状态的线性转换。
    # implementation=1: 实现模式，1 或 2。 模式 1 将把它的操作结构化为更多的小的点积和加法操作， 而模式 2 将把它们分批到更少，更大的操作中。 这些模式在不同的硬件和不同的应用中具有不同的性能配置文件。
    # return_sequences=False: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
    # return_state=False: 布尔值。除了输出之外是否返回最后一个状态。
    # go_backwards=False: 布尔值 (默认 False)。 如果为 True，则向后处理输入序列并返回相反的序列。
    # stateful=False: 布尔值 (默认 False)。 如果为 True，则批次中索引 i 处的每个样品的最后状态 将用作下一批次中索引 i 样品的初始状态。
    # unroll=False: 布尔值 (默认 False)。 如果为 True，则网络将展开，否则将使用符号循环。 展开可以加速 RNN，但它往往会占用更多的内存。 展开只适用于短序列
