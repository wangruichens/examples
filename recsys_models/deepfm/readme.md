# DeepFM Performance Analysis

采用tf.distribute.MirroredStrategy()。2块1080ti GPU，优化input_fn后，GPU资源基本可以满载训练，大概 11 global_step/sec。

```angular2
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 430.09       Driver Version: 430.09       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:01:00.0 Off |                  N/A |
| 55%   74C    P2   145W / 250W |  10736MiB / 11178MiB |     84%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 108...  Off  | 00000000:02:00.0 Off |                  N/A |
| 50%   69C    P2   130W / 250W |  10747MiB / 11178MiB |     82%      Default |
+-------------------------------+----------------------+----------------------+
```


- 两块GPU
```angular2
INFO:tensorflow:loss = 0.37087572, step = 30 (0.877 sec)
INFO:tensorflow:global_step/sec: 11.4237
INFO:tensorflow:loss = 0.35228056, step = 40 (0.875 sec)
INFO:tensorflow:global_step/sec: 11.5256
INFO:tensorflow:loss = 0.3553083, step = 50 (0.868 sec)
INFO:tensorflow:global_step/sec: 11.6155
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

