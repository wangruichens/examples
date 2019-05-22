# TF Serving 介绍、部署和Demo
---

tf serving:
支持模型热更新
支持版本管理
扩展性较好
稳定性，性能较好

### 一般工作流：

1、hdfs上的数据，使用spark/mapreduce/hive 进行数据分析和预处理

2、sub sample一部分数据，选择一个模型，预训练初始参数，交叉验证

3、使用全部数据集，spark to tfrecord 使用单机读取hdfs数据训练 or 多机多卡分布式训练

4、serving the model


# 一些解决方案：

## 方案1： yarn 3.1+ ： 
---
可以支持docker_image, [还不能提供稳定性保障](https://hadoop.apache.org/docs/r3.1.1/hadoop-yarn/hadoop-yarn-site/DockerContainers.html)

![image](https://github.com/wangruichens/samples/blob/master/distribute/tf/serving/serving.png)

[Docker+GPU support + tf serving + hadoop 3.1](https://community.hortonworks.com/articles/231660/tensorflow-serving-function-as-a-service-faas-with.html)


## 方案2： 模型Serving & 同步 from 美团blog
---
[参考链接](https://gitbook.cn/books/5b3adc411166b9562e9af3f6/index.html)

### 训练：tfrecord存放在hdfs上
### 预测：线上预估方案：

- 模型同步

我们开发了一个高可用的同步组件：用户只需要提供线下训练好的模型的 HDFS 路径，该组件会自动同步到线上服务机器上。该组件基于 HTTPFS 实现，它是美团离线计算组提供的 HDFS 的 HTTP 方式访问接口。同步过程如下：

    同步前，检查模型 md5 文件，只有该文件更新了，才需要同步。
    同步时，随机链接 HTTPFS 机器并限制下载速度。
    同步后，校验模型文件 md5 值并备份旧模型。
    
同步过程中，如果发生错误或者超时，都会触发报警并重试。依赖这一组件，我们实现了在 2min 内可靠的将模型文件同步到线上。

- 模型计算

主要的问题在于解决网络IO和计算性能。

    并发请求。一个请求会召回很多符合条件的广告。在客户端多个广告并发请求 TF Serving，可以有效降低整体预估时延。
    特征 ID 化。通过将字符串类型的特征名哈希到 64 位整型空间，可以有效减少传输的数据量，降低使用的带宽。
    定制的模型计算，针对性优化



## 方案3： Centos 7 + docker + tfserving (当前使用方案)

### 训练： 实现细节在[这里](https://github.com/wangruichens/samples/tree/master/distribute/tf/spark_tfrecord)

### 预测：线上预估方案：

1、prerequisit： 安装docker

使用tfserving的docker版本。不想去踩编译和GPU功能拓展的坑。 
```
# 1: 安装相关软件
sudo yum install -y yum-utils device-mapper-persistent-data lvm2
# 2: 添加软件源信息
sudo yum-config-manager --add-repo http://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo
# 3: 更新并安装 Docker-CE
sudo yum makecache fast
sudo yum -y install docker-ce
# 4: 开启Docker服务
sudo service docker start
# 5: 关闭Docker服务
sudo service docker stop
```

2、使用训练好的model, 使用hdfs tfrecord数据训练的手写数字识别model. 具体可以参考[这里](https://github.com/wangruichens/samples/tree/master/distribute/tf/spark_tfrecord)

模型很简单，参数量大概138w.

![image](https://github.com/wangruichens/samples/blob/master/distribute/tf/serving/model_des.png)


# Centos 7 + tf serving + GPU without Docker

愿意踩坑的可以自己使用bazel编译：[参考链接](https://www.dearcodes.com/index.php/archives/25/)

