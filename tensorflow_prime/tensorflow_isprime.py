import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())
print(tf.__version__)

def _odd_iter():
    n = 1
    while True:
        n = n + 2
        yield n

def _not_divisible(n):
    return  lambda x:x%n>0

def primes():
    yield 2
    it=_odd_iter()
    while True:
        n=next(it)
        yield n
        it=filter(_not_divisible(n),it)

def to_bin(x,bins=20):
    str=bin(x)[2:].zfill(bins)
    return [int(b) for b in str]

pos=[]
lens=300000
for n in primes():
    if n<lens:
       pos.append(n)
    else:
        break


n=2
p=pos.pop(0)
data_x=[]
data_y=[]
primenum=0
while n <lens:
    if n < p:
        # if all input data are not even, then the DNN can not find any pattern
        if np.random.uniform(0, 1) > 0.8 and n%2 !=0 :
            data_x.append(to_bin(n))
            data_y.append(0.0)
    if n==p:
        primenum+=1
        data_x.append(to_bin(n))
        data_y.append(1.0)
        if len(pos)>0:
            p=pos.pop(0)
        else:
            p=lens+1

    n+=1

# Another case : determin odd or even
# n=2
# data_x=[]
# data_y=[]
# primenum=0
# while n <lens:
#     if n % 2 ==0:
#         data_x.append(to_bin(n))
#         data_y.append(0.0)
#         primenum+=1
#     else:
#         data_x.append(to_bin(n))
#         data_y.append(1.0)
#     n+=1


print('positive cases num:',primenum)
print('total cases :',len(data_x))

from sklearn.model_selection import train_test_split

data_x=np.array(data_x)
data_y=np.array(data_y)
# data_y=np.expand_dims(np.array(data_y),axis=1)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=None,shuffle=True)

y_train=tf.keras.utils.to_categorical(y_train,num_classes=2)
y_test=tf.keras.utils.to_categorical(y_test,num_classes=2)

print('training dataset x:' ,x_train.shape)
print('training dataset y:' ,y_train.shape)

print('testing dataset x:' ,x_test.shape)
print('testing dataset y:' ,y_test.shape)
for i in range(10):
    print(x_test[i],y_test[i])

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# Should not use dropout here.
with tf.device('/cpu:0'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(30,activation='relu',input_shape=(20,)))
    # model.add(tf.keras.layers.Dropout(0.5))
    #
    model.add(tf.keras.layers.Dense(15,activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    #
    model.add(tf.keras.layers.Dense(5,activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(2, activation='softmax'))


# Multi gpu is very easy using keras
# model = tf.keras.utils.multi_gpu_model(model, gpus=2)


model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train, epochs=50, batch_size=2048)

acc= model.evaluate(x_test, y_test, batch_size=2048)
print('testing acc:',acc[1])

# Conclusion : IMPOSSIBLE to use DNN to predict prime ... of course
val_x=[300009,500005,333333,666666,499957,499969,499973,499979]
val=[]
for x in val_x:
    val.append(to_bin(x))
val=np.array(val)

res=model.predict(val)
for x,y in zip(val_x,res):
    print(x,' probability of prime:',y[1])
