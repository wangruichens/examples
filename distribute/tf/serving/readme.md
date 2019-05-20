# TFX tfserving 学习和部署
![image](https://github.com/wangruichens/samples/blob/master/distribute/tf/serving/arch.png)

train model and save ->running service -> make request

tf serving:
支持模型热更新
支持版本管理
扩展性较好
稳定性，性能较好

### 一般工作流：

1、hdfs上的数据，使用spark/mapreduce/hive 进行数据分析和预处理

2、sub sample一部分数据，选择一个模型，预训练初始参数，交叉验证

3、使用全部数据集，分布式训练， 多机多卡

4、serving the model

### yarn 3.0 目标：
- packaging
- gpu isolation
    - 多线程容易OOM
- easy shared FS(hdfs) access
- job tracking
- easy to deploy


![image](https://github.com/wangruichens/samples/blob/master/distribute/tf/serving/serving.png)
[Docker+GPU support + tf serving + hadoop 3.1](https://community.hortonworks.com/articles/231660/tensorflow-serving-function-as-a-service-faas-with.html)