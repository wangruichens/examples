
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

pos=[]
lens=100000
for n in primes():
    if n<lens:
       pos.append(n)
    else:
        break
print(pos[:10])
print(len(pos))


n=2
p=pos.pop(0)
train_x=[]
train_y=[]
while n <lens:
    if n < p:
        train_x.append(n)
        train_y.append(0)
    if n==p:
        train_x.append(n)
        train_y.append(1)
        if len(pos)>0:
            p=pos.pop(0)
        else:
            p=lens+1
    n+=1


from sklearn.model_selection import train_test_split

train_x=np.expand_dims(np.array(train_x),axis=1)
train_y=np.expand_dims(np.array(train_y),axis=1)

x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.1, random_state=42,shuffle=True)

print(x_train.shape)
print(x_test.shape)

with tf.device('/cpu:0'):
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')])


# parallel_model = tf.keras.utils.multi_gpu_model(model, gpus=2)

model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=128)

acc= model.evaluate(x_test, y_test, batch_size=128)
print(acc)