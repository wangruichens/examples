### 常用命令
---
列出所有的容器 ID
```
docker ps -aq
```
停止所有的容器
```
docker stop $(docker ps -aq)
```
删除所有的容器
```
docker rm $(docker ps -aq)
```
删除所有的镜像
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

