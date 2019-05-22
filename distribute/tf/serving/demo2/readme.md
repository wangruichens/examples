train model and save ->running service -> make request

docker中运行：
参考 ： https://www.tensorflow.org/tfx/serving/docker#serving_with_docker


docker run -p 8500:8500 --mount type=bind,source=/home/wangrc/test_serving/mnist_model_for_serving,target=/models/mnist -e MODEL_NAME=mnist -t tensorflow/serving

文件夹目录：
mnist_model_for_serving/1/ saved_model.pb  variables 

saved_model_cli show --dir . --all

测试：
python mnist_client.py   --num_tests=1 --server=127.0.0.1:8500
sudo netstat -nap | grep 8501


curl http://localhost:8501/v1/models/mnist
