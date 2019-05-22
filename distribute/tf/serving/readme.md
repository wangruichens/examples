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

### yarn 3.1+ ： 
---
可以支持docker_image, [还不能提供稳定性保障](https://hadoop.apache.org/docs/r3.1.1/hadoop-yarn/hadoop-yarn-site/DockerContainers.html)

![image](https://github.com/wangruichens/samples/blob/master/distribute/tf/serving/serving.png)

[Docker+GPU support + tf serving + hadoop 3.1](https://community.hortonworks.com/articles/231660/tensorflow-serving-function-as-a-service-faas-with.html)


# 模型Serving & 同步 from 美团blog
---
[参考链接](https://gitbook.cn/books/5b3adc411166b9562e9af3f6/index.html)

tfrecord存放在hdfs上：

![image](https://github.com/wangruichens/samples/blob/master/distribute/tf/serving/meituan1.png)

线上预估方案：

![image](https://github.com/wangruichens/samples/blob/master/distribute/tf/serving/meituan2.png)



# Centos 7 + docker + tfserving (当前使用方案)

1、prerequisit： 安装docker 
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

[参考链接](https://www.dearcodes.com/index.php/archives/25/)