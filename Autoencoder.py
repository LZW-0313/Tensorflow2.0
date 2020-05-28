# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:09:45 2020

@author: lx
"""

###################################  自编码网络  #########################################
import os                                        #用于查看与修改当前数据读取路径
import tensorflow as tf
from tensorflow import keras   
from tensorflow.keras import layers                       
import pandas as pd                              
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

##载入数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()
x_train = X_train.reshape(-1,28*28)/255.0 #预处理
x_test = X_test.reshape(-1,28*28)/255.0

print(x_train.shape,'',y_train.shape)
print(x_test.shape,'',y_test.shape)

##搭建网络
inputs = layers.Input(shape=(x_train.shape[1],),name='inputs')   #输入层
hidden = layers.Dense(20,activation='relu')(inputs)              #隐藏层
logits = layers.Dense(20,activation=None)(hidden)                #输出层
outputs = tf.sigmoid(logits,name='outputs')                      #输出层归一化
targets = layers.Dense(x_train.shape[1],name='targets')(outputs) #目标层
auto_encoder = keras.Model(inputs,targets)

##查看模型
auto_encoder.summary()

##设置loss与算法
auto_encoder.compile(optimizer='adam',loss='mse')

##开始训练！！！
history = auto_encoder.fit(x_train,x_train,batch_size=64,epochs=100,validation_split=0.1)

##可视化学习曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,0.1)
    plt.show()
plot_learning_curves(history)

##模型评估
auto_encoder.evaluate(x_test,x_test)

##预测
prediction = auto_encoder.predict(x_test)

##可视化预测结果
n=5
for i in range(n):
 ax = plt.subplot(2,n,i+1)
 plt.imshow(x_test[i].reshape(28,28))         #压缩且恢复后的图片
 plt.gray()
 ax.get_xaxis().set_visible(False)
 ax.get_yaxis().set_visible(False)
 
 ax = plt.subplot(2,n,n+i+1)
 plt.imshow(prediction[i].reshape(28,28))     #对应原图像
 plt.gray()
 ax.get_xaxis().set_visible(False)
 ax.get_yaxis().set_visible(False)
plt.show()