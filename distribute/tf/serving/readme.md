# TFX tfserving 学习和部署
---
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

### yarn 3.0+ 目标：
- packaging
- gpu isolation
    - 多线程容易OOM
- easy shared FS(hdfs) access
- job tracking
- easy to deploy


![image](https://github.com/wangruichens/samples/blob/master/distribute/tf/serving/serving.png)
[Docker+GPU support + tf serving + hadoop 3.1](https://community.hortonworks.com/articles/231660/tensorflow-serving-function-as-a-service-faas-with.html)


# 模型同步 from 美团blog
---
我们开发了一个高可用的同步组件：用户只需要提供线下训练好的模型的 HDFS 路径，该组件会自动同步到线上服务机器上。该组件基于 HTTPFS 实现，它是美团离线计算组提供的 HDFS 的 HTTP 方式访问接口。同步过程如下：

- 同步前，检查模型 md5 文件，只有该文件更新了，才需要同步。
- 同步时，随机链接 HTTPFS 机器并限制下载速度。
- 同步后，校验模型文件 md5 值并备份旧模型。

同步过程中，如果发生错误或者超时，都会触发报警并重试。依赖这一组件，我们实现了在 2min 内可靠的将模型文件同步到线上。

