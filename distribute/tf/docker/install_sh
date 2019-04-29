---
docker 设置代理
Environment="HTTPS_PROXY=socks5://127.0.0.1:1080"
---
curl 代理下载 kubectl
curl --socks5-hostname localhost:1080 -LO https://storage.googleapis.com/kubernetes-release/release/v1.14.0/bin/linux/amd64/kubectl
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin/kubectl
---
curl 代理下载 minikube
curl --socks5-hostname localhost:1080 -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 \
  && chmod +x minikube
sudo cp minikube /usr/local/bin && rm minikube


--- socks5 代理映射https

privoxy 的配置如下 sudo vim /etc/privoxy/config
listen-address 0.0.0.0:1081
forward-socks5 / 127.0.0.1:1080 .
sudo systemctl restart privoxy

修改~/.bashrc
export http_proxy=http://127.0.0.1:1081
export https_proxy=https://127.0.0.1:1081

source ~/.bashrc

测试一下 privoxy 是否正常工作

# check process
ps -ef | grep privoxy
# curl port
curl http://127.0.0.0:1081

(fail 在虚拟机里面仍然不能curl http://10.0.2.2:1081 )

---
下载 iso
minikube start
minikube start --docker-env HTTP_PROXY=http://localhost:1081 \ --docker-env HTTPS_PROXY=https://localhost:1081
minikube start --docker-env HTTP_PROXY=http://10.0.2.2:1081 --docker-env HTTPS_PROXY=https://10.0.2.2:1081

---
国内阿里minikube
curl -Lo minikube http://kubernetes.oss-cn-hangzhou.aliyuncs.com/minikube/releases/v1.0.0/minikube-linux-amd64 && chmod +x minikube && sudo mv minikube /usr/local/bin/