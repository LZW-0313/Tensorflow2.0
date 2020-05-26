# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:15:57 2020

@author: lx
"""
import os                                        #用于查看与修改当前数据读取路径
import tensorflow as tf
from tensorflow import keras                          
import pandas as pd                              
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #用于数据归一化

################################### 简单线性回归 #########################################
print("Tensorflow version:{}".format(tf.__version__)) #查看当前Tensorflow版本
os.getcwd()
os.chdir('C:\\Users\\lx\\Desktop')       ##更改路径至桌面
data = pd.read_csv("data.csv")           ##导入数据
plt.scatter(data.x,data.y)  ##可视化
 
x = data.x
y = data.y
 
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape=(1,)))
model.summary()
 
model.compile(optimizer="adam",   ##优化方法使用内置的adam
              loss="mse"          ##损失函数使用均方差  
)
 
 
history = model.fit(x,y,epochs=5000) ##对构建的模型进行训练;  训练5000次

def plot_learning_curves(history):   ##可视化学习曲线
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,10)
    plt.show()
plot_learning_curves(history)

model.predict(x)
 
model.predict(pd.Series(20))        ##输入值也可为Series类型

################################## DNN回归 ####################################
data=pd.read_csv("data2.csv")
train=data[0:8] #分为训练集
CV=data[8:9]   #验证集(进行模型选择)
test=data[9:10] 
train_x=train[['a','b','c','d','e']]
train_y=train.y
CV_x=CV[['a','b','c','d','e']]
CV_y=CV.y
test_x=test[['a','b','c','d','e']]
test_y=test.y
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(2, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1),
])
# model.summary 可以显示出网络结构的具体信息
model.summary()
# 编译模型， 损失函数为均方误差函数，优化函数为随机梯度下降
model.compile(loss="mean_squared_error", optimizer = tf.keras.optimizers.SGD(0.001))
# 回调函数使用了EarlyStopping，patience设为5， 阈值设置为1e-2
callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
#训练模型
history = model.fit(train_x,train_y,validation_data=(CV_x,CV_y),epochs = 100,callbacks=callbacks)
#可视化学习曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,100)
    plt.show()
plot_learning_curves(history)
#在测试集上的表现
model.evaluate(test_x,test_y)
#预测
model.predict(test_x[1:2])

##################################### logistic\DNN 二分类 ######################
data=pd.read_csv("data3.csv")
train=data[0:10] #分为训练集和测试集
test=data[10:16] 
train_x=train[['a','b','c','d','e']]
train_y=train.y
test_x=test[['a','b','c','d','e']]
test_y=test.y
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(5,)),  ##也可扩充多层（每层为'ReLu',最后一层为logistic激活函数）
])
# model.summary 可以显示出网络结构的具体信息
model.summary()
# 编译模型， 损失函数为均方误差函数，优化函数为随机梯度下降
model.compile(loss="binary_crossentropy", 
              optimizer = tf.keras.optimizers.SGD(0.001),
              metrics=['acc'])
# 提前终止 (回调函数使用了EarlyStopping，patience设为5， 阈值设置为1e-2)
callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
#训练模型
history = model.fit(train_x,train_y,validation_data=(test_x,test_y),epochs = 100,callbacks=callbacks)
#可视化学习曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
plot_learning_curves(history)

################################### DNN 多分类 ###############################
data=pd.read_csv("data4.csv")
train=data[0:6] #分为训练集
CV=data[6:9]   #验证集(进行模型选择)
test=data[9:11] 
train_x=train[['a','b','c']]
train_y=train.y
CV_x=CV[['a','b','c']]
CV_y=CV.y
test_x=test[['a','b','c']]
test_y=test.y
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(2, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(3,activation='softmax') ##注意输出层结点数为lable种类数
])
# model.summary 可以显示出网络结构的具体信息
model.summary()
# 编译模型， 损失函数为均方误差函数，优化函数为随机梯度下降
model.compile(loss="sparse_categorical_crossentropy", #若lable为one-hot 编码：[0, 0, 1], [1, 0, 0], [0, 1, 0]，则选用"categorical_crossentropy"
              optimizer = "adam",
              metrics=['acc'])
#训练模型
history = model.fit(train_x,train_y,validation_data=(CV_x,CV_y),epochs = 100)
#可视化学习曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,2)
    plt.show()
plot_learning_curves(history)
#在测试集上的表现
model.evaluate(test_x,test_y)
#预测
model.predict(test_x[0:1])

########################### 基于DNN的Fashion数据集分类 ##############################
#数据准备
mnist=tf.keras.datasets.fashion_mnist
(training_images,training_labels),(test_images,test_labels)=mnist.load_data()
plt.imshow(training_images[11])      #可视化单张图片
print(training_labels[11])
print(training_images[11])
np.array(training_images[11]).shape  #查看单张图片的像素点个数
training_images=training_images/255.0#归一化处理
test_images=test_images/255.0
#构建神经网络模型结构 (此例构建了一个单隐层FNN)
model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(512,activation="relu"),
    tf.keras.layers.Dense(10,activation="softmax")
])
# model.summary 可以显示出网络结构的具体信息
model.summary()
#可视化网络结构
tf.keras.utils.plot_model(model)   #目前还实现不了！！需要去官网下包,放到相应环境变量里！！
#设置loss与优化算法
model.compile(loss="sparse_categorical_crossentropy",
              optimizer = "adam",
              metrics=['acc'])
#给算法设置提前终止 (回调函数使用了EarlyStopping，patience设为5， 阈值设置为1e-2)
callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
#开始训练！！！
history = model.fit(training_images,training_labels,
                    validation_data=(test_images,test_labels), #这里把测试集也当作验证集用于模型选择(选择网络层数、每层节点数等超参数)
                    epochs = 10,#epochs(训练周期)指的就是训练过程中全体数据将被轮流训练多少次;若不设置batch_size,会自动配置其大小
                    callbacks=callbacks) 
#可视化学习曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
plot_learning_curves(history)
#在测试集上的表现
model.evaluate(test_images,test_labels)

######################## 基于CNN的Fashion数据集分类 #############################
##数据准备
fashion_mnist = tf.keras.datasets.fashion_mnist # 该数据集是黑白服装数据集
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()#拆分训练集和测试集
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]#再次将训练集拆分为训练集和验证集
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]
print(x_train[0].dtype)
print(x_train[0]) # 是一个数据矩阵 28*28, 矩阵中的每一个数值都是uint8类型
print(y_train[0]) #这里的y值均为数字编码，非向量，所以后面定义模型损失函数为 sparse_categorical_crossentropy
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

#展示单张图片
def show_single_image(img_arr):
    plt.imshow(img_arr, cmap="binary") #cmap:将标准化标量映射为颜色, binary代表白底黑字
    plt.show()
show_single_image(x_train[0])

#展示图片组
def show_imgs(n_rows, n_cols, x_data, y_data, class_names):
    assert len(x_data) == len(y_data)
    assert n_rows * n_cols < len(x_data)
    plt.figure(figsize = (n_cols * 1.4, n_rows * 1.6)) #.figure 在plt中绘制一张图片
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1) # 创建单个子图
            plt.imshow(x_data[index], cmap="binary", interpolation='nearest')
            plt.axis('off') #取消坐标系
            plt.title(class_names[y_data[index]]) #标题
    plt.show()
    
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
show_imgs(3, 5, x_train, y_train, class_names)

#数据升阶,再加一阶通道,为了适合下面的CNN结构
x_train=x_train.reshape(-1,28,28,1) 
x_valid=x_valid.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)


#归一化处理
print(np.max(x_train), np.min(x_train)) #查看值域
x_train=x_train/255.0
x_valid=x_valid/255.0
x_test=x_test/255.0


##构建神经网络模型结构 (此例构建了一个CNN)
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',
                           input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),                          
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation="relu"),
    tf.keras.layers.Dense(10,activation="softmax")
])
# model.summary 可以显示出网络结构的具体信息
model.summary()

##设置loss与优化算法
model.compile(loss="sparse_categorical_crossentropy",
              optimizer = "adam",
              metrics=['acc'])

##给算法设置提前终止 (回调函数使用了EarlyStopping，patience设为5， 阈值设置为1e-2)
callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]

##开始训练！！！
history = model.fit(x_train,y_train,
                    validation_data=(x_valid,y_valid), 
                    epochs = 10,#epochs(训练周期)指的就是训练过程中全体数据将被轮流训练多少次;若不设置batch_size,自动配置其大小
                    callbacks=callbacks) 
##可视化学习曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
plot_learning_curves(history)

##在测试集上的表现
model.evaluate(x_test,y_test)


######################### 自然语言处理——————利用 word2vec网络进行语义情感分析 ############################
##数据准备
os.getcwd()                        #查看文件读取路径
os.chdir('C:\\Users\\lx\\Desktop') #更改路径至桌面
vocab_size = 10000
(train_x, train_y), (test_x, text_y) = keras.datasets.imdb.load_data(num_words=vocab_size) #载入数据(加载太慢...)
print(train_x[0])  #查看每一条评论
print(train_x[1])

##其他尝试读取imdb数据集的方法(numpy法)
os.getcwd()                                                      #查看文件读取路径
os.chdir('C:\\Users\\lx\\Desktop')                               #更改路径至桌面
vocab_size = 100000
data = np.load("C:/Users/lx/Desktop/imdb.npz",allow_pickle=True) #载入数据(先下下来)
data.files
train_x=data['x_train']
train_y=data['y_train']
test_x=data['x_test']
test_y=data['y_test']
print(train_x[0])
print(train_x[1])
print(train_y[0])

word_index = keras.datasets.imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3
reverse_word_index = {v:k for k, v in word_index.items()}
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
print(decode_review(train_x[0]))

maxlen = 500
train_x = keras.preprocessing.sequence.pad_sequences(train_x,value=word_index['<PAD>'],
                                                    padding='post', maxlen=maxlen)
test_x = keras.preprocessing.sequence.pad_sequences(test_x,value=word_index['<PAD>'],
                                                    padding='post', maxlen=maxlen)

##构建网络框架
embedding_dim = 16
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
    
])
model.summary()

##设置loss与优化算法
model.compile(optimizer=keras.optimizers.Adam(),
             loss=keras.losses.BinaryCrossentropy(),
             metrics=['accuracy'])

##开始训练!!
history = model.fit(train_x, train_y, epochs=10, batch_size=512, validation_split=0.1) #0.1指交叉验证集占训练集的1/10！

##可视化学习曲线
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.figure(figsize=(16,9))
plt.show()

##查看在测试集上的表现
model.evaluate(test_x,test_y)

##查看某层权值shape
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)

##放到 Embedding Projector 上进行可视化
out_v = open('vecs.tsv', 'w')
out_m = open('meta.tsv', 'w')
for word_num in range(vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

##########################  利用 DNN 做时间序列预测 #############################
##构建相关函数
def plot_series(time,series,format="-",start=0,end=None):
    plt.plot(time[start:end],series[start:end],format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    
def trend(time,slope=0):
    return slope*time

def seasonal_pattern(season_time):
    """Iust an arbitrary pattern,you can change it if you wish"""
    return np.where(season_time<0.4,
                    np.cos(season_time*2*np.pi),
                    1/np.exp(3*season_time))

def seasonality(time,period,amplitude=1,phase=0):
    """Repeats the same pattern at each period"""
    season_time=((time+phase)%period)/period
    return amplitude*seasonal_pattern(season_time)

def noise(time,noise_level=1,seed=None):
    rnd=np.random.RandomState(seed)
    return rnd.randn(len(time))*noise_level

##设置相关超参数
time=np.arange(4*365+1,dtype="float32")
baseline=10
series=trend(time,0.1)
amplitude=40
slope=0.05
noise_level=5

##创建时间序列
series=baseline+trend(time,slope)+seasonality(time,period=365,amplitude=amplitude)

##加入噪声
series += noise(time,noise_level,seed=42)

##分为训练集和验证集(测试集)
split_time = 1000
time_train=time[:split_time]
x_train=series[:split_time]
time_test=time[split_time:]
x_test=series[split_time:]

window_size=20
batch_size=32
shuffle_buffer_size=1000

def windowed_dataset(series,window_size,batch_size,shuffle_buffer):
    dataset=tf.data.Dataset.from_tensor_slices(series)
    dataset=dataset.window(window_size+1,shift=1,drop_remainder=True)
    dataset=dataset.flat_map(lambda window:window.batch(window_size+1))
    dataset=dataset.shuffle(shuffle_buffer).map(lambda window:(window[:-1],window[-1]))
    dataset=dataset.batch(batch_size).prefetch(1)
    return dataset

##构建合适的数据结构
dataset = windowed_dataset(x_train,window_size,batch_size,shuffle_buffer_size)

##构建DNN模型框架（此例为两隐层的DNN）
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10,input_shape=[window_size],activation="relu"),
    tf.keras.layers.Dense(10,activation="relu"),
    tf.keras.layers.Dense(1)
])

##查看模型框架
model.summary()

##选择loss与优化算法
model.compile(loss="mse",optimizer=tf.keras.optimizers.SGD(lr=8e-6,momentum=0.9))

##开始训练！！！
history = model.fit(dataset,epochs=100)

##可视化学习曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(30,40)
    plt.show()
plot_learning_curves(history)

##预测(模型在测试集上的表现)
forecast = []
for time in range(len(series)-window_size):
    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:,0,0]

plt.figure(figsize=(10,6))

plot_series(time_test,x_test)  #在测试集上可视化模型预测结果
plot_series(time_test,results)

tf.keras.metrics.mean_absolute_error(x_test,results).numpy() #查看在测试集上的MAE误差

########################  利用 RNN\LSTM 做时间序列预测 ##########################
##数据准备
train_set= windowed_dataset(x_train,window_size,batch_size=128,
                            shuffle_buffer=shuffle_buffer_size)
##构建RNN模型框架
model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x:tf.expand_dims(x,axis=-1),input_shape=[None]),
    tf.keras.layers.SimpleRNN(40,return_sequences=True),
    tf.keras.layers.SimpleRNN(40),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x:x*100.0)
])

##或构建LSTM模型框架
model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x:tf.expand_dims(x,axis=-1),input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x:x*100.0)
])
    
##查看模型
model.summary()

##每epoch自动调整步长
Ir_schedule=tf.keras.callbacks.LearningRateScheduler(lambda epoch:1e-8*10**(epoch/20)) 

##设置loss与算法
optimizer = tf.keras.optimizers.SGD(lr=1e-8,momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),optimizer=optimizer,metrics=["mae"])

##开始训练！！！
history = model.fit(train_set,epochs=100,callbacks=[Ir_schedule])

##查看学习速率的变化对loss的影响，选择最佳步长
plt.semilogx(history.history["lr"],history.history["loss"])
plt.axis([1e-8,1e-4,0,30])

##调整步长后重新训练
optimizer = tf.keras.optimizers.SGD(lr=1e-5,momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),optimizer=optimizer,metrics=["mae"])
history = model.fit(train_set,epochs=100)

##可视化学习曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,30)
    plt.show()
plot_learning_curves(history)

##预测(模型在测试集上的表现)
forecast = []
for time in range(len(series)-window_size):
    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:,0,0]

plt.figure(figsize=(10,6))

plot_series(time_test,x_test)  #在测试集上可视化模型预测结果
plot_series(time_test,results)

tf.keras.metrics.mean_absolute_error(x_test,results).numpy() #查看在测试集上的MAE误差


