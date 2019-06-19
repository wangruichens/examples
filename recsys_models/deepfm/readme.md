# DeepFM Performance Analysis

采用tf.distribute.MirroredStrategy()。2块1080ti GPU，大概 9 global_step/sec。

```angular2
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 430.09       Driver Version: 430.09       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:01:00.0 Off |                  N/A |
| 44%   62C    P2   106W / 250W |  10777MiB / 11178MiB |     80%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 108...  Off  | 00000000:02:00.0 Off |                  N/A |
| 35%   53C    P2   144W / 250W |  10747MiB / 11178MiB |     66%      Default |
+-------------------------------+----------------------+----------------------+
```


- 两块GPU
```angular2
INFO:tensorflow:loss = 0.41606092, step = 30 (0.964 sec)
INFO:tensorflow:global_step/sec: 9.57605
INFO:tensorflow:loss = 0.3935175, step = 40 (1.044 sec)
INFO:tensorflow:global_step/sec: 9.48205
INFO:tensorflow:loss = 0.38121438, step = 50 (1.055 sec)
INFO:tensorflow:global_step/sec: 9.57724
```

- 一块GPU
```angular2
INFO:tensorflow:loss = 0.44679362, step = 20 (1.432 sec)
INFO:tensorflow:global_step/sec: 4.93999
INFO:tensorflow:loss = 0.4266003, step = 30 (2.024 sec)
INFO:tensorflow:global_step/sec: 4.809
INFO:tensorflow:loss = 0.3959203, step = 40 (2.079 sec)
INFO:tensorflow:global_step/sec: 4.98097
```

