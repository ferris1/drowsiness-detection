# -*- coding: UTF-8 -*-
# write by feng
import keras
import numpy as np
from keras.models import Sequential
import tensorflow
import pandas as pd
from keras.layers import Dense, Activation,Dropout
from keras import optimizers
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras.utils import plot_model
from keras.callbacks import TensorBoard
import time 
import common
epochs = 300
batch_size = 128
#建模
"""
model = Sequential()
model.add(Dense(30,input_dim=common.VECTOR_SIZE))
model.add(Activation('relu'))
model.add(Dropout(0.0))
model.add(Dense(30,input_dim=30))
model.add(Activation('relu'))
model.add(Dropout(0))
model.add(Dense(30,input_dim=30))
model.add(Activation('relu'))
model.add(Dropout(0.0))

model.add(Dense(30,input_dim=30))
model.add(Activation('relu'))
model.add(Dropout(0))

model.add(Dense(1,input_dim=30))
model.add(Activation('sigmoid'))
#存储模型图
plot_model(model, to_file='./data/model.png')
"""
#加载模型
model = load_model(common.model_file)

dataframe = pd.read_csv(common.data_file,header=0)
#dataframe = dataframe.sample(frac=1)#打乱
#print(dataframe.head())
#print(dataframe.shape)
dataset = dataframe.values
print(dataset)
train_X = dataset[:,:-1]
train_Y = dataset[:,-1]

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer="Adam", #Adam RMSprop
        loss='binary_crossentropy',
        metrics=['accuracy'])
#可视化
tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#               batch_size=32,     # 用多大量的数据计算直方图
                write_graph=True,  # 是否存储网络结构图
                write_grads=True, # 是否可视化梯度直方图
                write_images=True,# 是否可视化参数
                embeddings_freq=0,
                embeddings_layer_names=None,
                embeddings_metadata=None)

#转换成int型
#encoder = LabelEncoder()
#encoded_Y = encoder.fit_transform(train_Y)
#encode_y = keras.utils.to_categorical(train_Y, num_classes=3)

#X_train, X_test, Y_train, Y_test = train_test_split(train_X, encode_y, test_size=0.3, random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_Y, test_size=0.3, random_state=0)
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
            verbose=1,
            validation_split=0.1,
            shuffle=True,
            callbacks=[tbCallBack])

score = model.evaluate(X_test,Y_test,batch_size=batch_size,verbose=1)
#print(X_test)
#print(Y_test)

model.save(common.model_file)  # 创建 HDF5 文件 'my_model.h5'
print("success save to {}".format(common.model_file))

print('Test score:', score[0])
print('Test accuracy:', score[1])


