```
# For gRpc，默认端口8500
docker run -p 8500:8500 --mount type=bind,source=/home/wangrc/test_serving/mnist_model_for_serving,target=/models/mnist -e MODEL_NAME=mnist -t tensorflow/serving

# For REST, 默认端口8501
docker run -p 8501:8501 --mount type=bind,source=/home/wangrc/mnist_model,target=/models/mnist -e MODEL_NAME=mnist -t tensorflow/serving
```


```
tensorflow_model_server --port=8500 --rest_api_port=8501 \
  --model_name=my_model --model_base_path=/models/my_model
```

