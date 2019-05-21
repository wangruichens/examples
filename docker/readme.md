### 常用命令
---
- 列出所有的容器 ID
```
docker ps -aq
```
- 停止所有的容器
```
docker stop $(docker ps -aq)
```
- 删除所有的容器
```
docker rm $(docker ps -aq)
```
- 删除所有的镜像
```
docker rmi $(docker images -q)
```

### [为docker 设置代理](https://docs.docker.com/config/daemon/systemd/)
---
文件位置： /etc/systemd/system/docker.service.d/http-proxy.conf

```
[Service]
Environment="HTTP_PROXY=socks5://127.0.0.1:1080"
```
文件位置： /etc/systemd/system/docker.service.d/https-proxy.conf
```
[Service]
Environment="HTTPS_PROXY=socks5://127.0.0.1:1080"
```

Docker Swarm 和 Docker Compose 一样，都是 Docker 官方容器编排项目，但不同的是，Docker Compose 是一个在单个服务器或主机上创建多个容器的工具，而 Docker Swarm 则可以在多个服务器或主机上创建容器集群服务，对于微服务的部署，显然 Docker Swarm 会更加适合。

# Docker Machine
```
base=https://github.com/docker/machine/releases/download/v0.16.0 &&
  curl -L $base/docker-machine-$(uname -s)-$(uname -m) >/tmp/docker-machine &&
  sudo install /tmp/docker-machine /usr/local/bin/docker-machine
```


