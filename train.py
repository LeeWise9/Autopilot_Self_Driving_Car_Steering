# -*- coding: utf-8 -*-
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
#控制显存占用：动态增长
'''
import csv
import cv2
import glob
import json
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from keras import callbacks
from keras.optimizers import Adam
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, PReLU
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def horizontal_flip(img, degree):
    #按照50%的概率水平翻转图像
    choice = np.random.choice([0, 1])
    if choice == 1:
        img, degree = cv2.flip(img, 1), -degree
    return (img, degree)

def random_brightness(img, degree):
    #随机调整输入图像的亮度，调整强度于 0.1(变黑)和1(无变化)之间
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    #调整亮度 V
    alpha = np.random.uniform(low=0.1,high=1.0,size=None)
    v = hsv[:,:,2]
    v = v * alpha
    hsv[:,:,2] = v.astype('uint8')
    rgb = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)
    return (rgb, degree)

def left_right_random_swap(img_address, degree, degree_corr=0.25):
    #随机从左， 中， 右图像中选择一张图像， 并相应调整转动的角度
    swap = np.random.choice(['L','R','C'])
    if swap == 'L':
        img_address = img_address.replace('center','left')
        corrected_degree = np.arctan(math.tan(degree) + degree_corr)
        return (img_address, corrected_degree)
    elif swap == 'R':
        img_address = img_address.replace('center','right')
        corrected_degree = np.arctan(math.tan(degree) - degree_corr)
        return (img_address, corrected_degree)
    else:
        return (img_address, degree)
    
def image_transformation(img_address, degree, data_dir):
    #调用上述函数完成图像预处理：选图、调亮度、水平翻转
    img_address, degree = left_right_random_swap(img_address, degree)
    img = cv2.imread(data_dir + img_address)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, degree = random_brightness(img, degree)
    img, degree = horizontal_flip(img, degree)
    return (img, degree)

def discard_zero_steering(degrees, rate):
    #从角度为零的index中随机选择部分index返回，丢弃
    steering_zero_idx = np.where(degrees == 0)[0]
    size_del = int(len(steering_zero_idx) * rate)
    #选出size_del个元素丢弃:
    return np.random.choice(steering_zero_idx, size=size_del, replace=False)

def get_model(shape):
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(1, 1), padding="same", activation='relu', input_shape=shape))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer=Adam(lr=1e-3), loss='mean_squared_error')
    return model

def batch_generator(x, y, batch_size, shape, training=True, data_dir='data/', 
                    monitor=True, yieldXY=True, discard_rate=0.8):
    """
    training: True产生训练数据，False产生validation数据
    monitor: 保存一个batch样本： 'X_batch_sample.npy'，'y_bag.npy’
    yieldXY: True返回(X, Y)，False返回 X 
    discard_rate: 随机丢弃角度为零的训练数据的概率
    """
    if training:
        y_bag = []
        x, y = shuffle(x, y)
        rand_zero_idx = discard_zero_steering(y, rate=discard_rate)
        new_x = np.delete(x, rand_zero_idx, axis=0)
        new_y = np.delete(y, rand_zero_idx, axis=0)
    else: # train要删数据，vail不删
        new_x = x
        new_y = y
    offset = 0 # 每完成一次循环增加 batch_size
    while True:
        X = np.empty((batch_size, *shape)) # *号连接 tuple和 int
        Y = np.empty((batch_size, 1)) # X存放图片，Y存放标签
        for example in range(batch_size):
            img_address, img_steering = new_x[example + offset], new_y[example + offset]
            if training:
                img, img_steering = image_transformation(img_address, img_steering, data_dir)
            else:
                img = cv2.imread(data_dir + img_address)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X[example,:,:,:] = cv2.resize(img[80:140, 0:320], (shape[0], shape[1]) ) / 255 - 0.5
            Y[example] = img_steering
            if monitor:
                y_bag.append(img_steering)
            
            #到达原来数据的结尾后, 从头开始
            if (example + 1) + offset > len(new_y) - 1:
                x, y = shuffle(x, y)
                rand_zero_idx = discard_zero_steering(y, rate=discard_rate)
                new_x = x
                new_y = y
                new_x = np.delete(new_x, rand_zero_idx, axis=0)
                new_y = np.delete(new_y, rand_zero_idx, axis=0)
                offset = 0
        if yieldXY:
            yield (X, Y)
        else:
            yield X
        offset = offset + batch_size
        if monitor:
            np.save('y_bag.npy', np.array(y_bag))
            np.save('Xbatch_sample.npy', X ) 


if __name__ == '__main__':
    SEED = 13
    data_path = 'data/'
    with open(data_path + 'driving_log.csv', 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        log = []
        for row in file_reader:
            log.append(row)

    log = np.array( log )
    # 去掉文件第一行
    log = log[1:,:] 
    
    # 判断图像文件数量是否等于csv日志文件中记录的数量
    ls_imgs = glob.glob(data_path+ 'IMG/*.jpg')
    assert len(ls_imgs) == len(log)*3, 'number of images does not match'

    # 使用20%的数据作为测试数据
    validation_ratio = 0.2
    shape = (128, 128, 3)
    batch_size = 128
    nb_epoch = 20

    x_ = log[:, 0]
    y_ = log[:, 3].astype(float)
    x_, y_ = shuffle(x_, y_)
    X_train, X_val, y_train, y_val = train_test_split(x_, y_, test_size=validation_ratio, random_state=SEED)

    print('batch size: {}'.format(batch_size))
    print('Train set size: {} | Validation set size: {}'.format(len(X_train), len(X_val)))
        
    samples_per_epoch = batch_size 
    # 使得validation数据量大小为batch_size的整数陪
    nb_val_samples = len(y_val) - len(y_val)%batch_size
    model = get_model(shape)
    print(model.summary())
    
    # 根据validation loss保存最优模型
    save_best = callbacks.ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, 
                                         save_best_only=True, mode='auto')

    # 如果训练持续没有validation loss的提升, 提前结束训练                                
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15, 
                                         verbose=0, mode='auto')
    callbacks_list = [early_stop, save_best]

    history = model.fit_generator(batch_generator(X_train, y_train, batch_size, shape, training=True, monitor=False),
                                  steps_per_epoch = samples_per_epoch,
                                  validation_steps = nb_val_samples // batch_size,
                                  validation_data = batch_generator(X_val, y_val, batch_size, shape, 
                                                                  training=False, monitor=False),
                                  epochs=nb_epoch, verbose=1, callbacks=callbacks_list)

    with open('./trainHistoryDict.p', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('train_val_loss.jpg',dpi=600)

    # 保存模型
    with open('model.json', 'w') as f:
            f.write( model.to_json() )
    model.save('model.h5')
    print('Done!')