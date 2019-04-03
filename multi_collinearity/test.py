import tensorflow as tf
from tensorflow.python.keras import layers,Model,callbacks,initializers,regularizers,activations
import  numpy as np
# tf.enable_eager_execution()



x=np.array([[20.5,19.8],[21.2,20.4],[22.8,21.1],
           [18.2,23.6],[20.3,24.9],[21.8,26.7],
           [25.2,28.9],[30.7,31.3],[36.1,35.8],
           [44.3,38.2]])

x1=[]
x2=[]
for i in x:
    x1.append(i[0])
    x2.append(i[1])


y=np.array([7.8,8.6,8.7,7.9,8.4,8.9,10.4,11.6,13.9,15.8])
print(x.shape)
print(y.shape)

with tf.device('/cpu:0'):
    inputs=layers.Input(shape=(2,))
    # x=layers.Dense(2)(inputs)
    out=layers.Dense(1)(inputs)
    model=Model(inputs=inputs,outputs=out)


dataset=tf.data.Dataset.from_tensor_slices((x,y)).batch(3).repeat()

# print(dataset.take(1))

model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001),loss='mean_squared_error')

print_weights = callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: print('\n',model.layers[1].get_weights()))


model.summary()
# model.fit(dataset,epochs=3,steps_per_epoch=20,callbacks=[print_weights])
model.fit(dataset,epochs=20,steps_per_epoch=1)

