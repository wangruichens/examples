import tempfile
import os
import subprocess

from hdfs import InsecureClient
import time
import sys



hdfs_path = 'hdfs://cluster/user/wangrc/mnist_model_for_serving'
root_path = "/home/wangrc"
c = InsecureClient(url="hdfs://cluster", user='wangrc', root=root_path)
hdfs_files = c.list('/user/wangrc', True)
for f in hdfs_files:
    print(f)
c.download('/user/root/pyhdfs/1.log', '.', True)


export_path='.'
cmd='nohup tensorflow_model_server \
  --rest_api_port=8256 \
  --model_name=mnist_model \
  --model_base_path="{0}" >server.log 2>&1'.format(export_path)
subprocess.call(cmd,shell=True)