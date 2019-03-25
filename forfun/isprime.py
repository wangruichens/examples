
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
lens=10000
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



