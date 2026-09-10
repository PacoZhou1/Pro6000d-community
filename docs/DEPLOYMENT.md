# R30 DMA9100部署

面向相同四卡6000D、Linux宿主、Docker和NVIDIA Container Toolkit。只固定R30镜像；勿将完整replacement文件套到R31/R32或主线。读SOURCE_LOCK与THIRD_PARTY_NOTICES，自己准备GLM-5.3-Flash-NVFP4权重。

## 创建容器

示例统一将仓库装到 `/opt/rtx6000d-glm53-lab`。模型目录自行设定，缓存默认 `/var/lib/glm53-lab/cache`。若本机已有服务，先安排停机窗口和保存旧容器；脚本拒绝覆盖同名容器。

```bash
git clone https://github.com/PacoZhou1/rtx6000d-glm53-lab.git
sudo cp -a rtx6000d-glm53-lab /opt/
cd /opt/rtx6000d-glm53-lab
sudo nvidia-smi -i 0,1,2,3 -pl 350
sudo env MODEL_DIR=/srv/models/GLM-5.3-Flash-NVFP4 bash deploy/create.sh
```

create只创建容器。首次启动会加载权重、编译/捕获图，需要等待health；不要把docker进程存在当模型ready。脚本保存实测参数，使用可从registry拉取的manifest digest；它与原机离线导入manifest不同，但image config/rootfs身份一致。

## 开机自启与保活

```bash
sudo install -m 644 deploy/nvidia-power-limit.service /etc/systemd/system/
sudo install -m 644 deploy/homelab-vllm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-power-limit.service
sudo systemctl enable --now homelab-vllm.service
sudo journalctl -u homelab-vllm -f
```

systemd是唯一重启控制方，Docker restart=no。每次启动校验四个patch SHA、权重config存在、四卡功率均350W。进程退出后10秒重试，600秒最多5次；不无限重启掩盖错误。Docker health显示异常本身不会结束进程，因此**这个便携模板不包含自动重启健康但卡死进程的外部watchdog**。原部署另有监控/flight recorder；未将私有看板和业务网关代码混入本实验仓库。

```bash
curl -f http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8000/v1/models
nvidia-smi --query-gpu=power.limit,power.draw,temperature.gpu --format=csv
systemctl is-enabled nvidia-power-limit homelab-vllm
```

当前原机已验证真实重启恢复。公开脚本的路径参数化和模板仅做静态/无GPUmock验证，没有为发仓库重启正在使用的生产模型。迁移后需自行完成真实启动、health、模型ID和最小推理请求验收。

## 网络与SSH

推理只监听127.0.0.1:8000。用受认证的网关反代OpenAI API，再接Caddy TLS；API key管理应在网关完成，不能裸露无鉴权8000或BMC到公网。常驻Mac Pro可以承担中转，示意为：外部客户端→TLS网关/跳板→GPU服务器；正式跑分在GPU本机执行。

局域网直连 `ssh GPU_USER@GPU_LAN_IP`；外网可用 `ssh -J MAC_USER@PUBLIC_HOST:SSH_PORT GPU_USER@GPU_LAN_IP`。实际地址和密钥不随公开仓库发布。网关、BMC账号与API key不是同一套凭据。

回退时停止新systemd单元并启动事先保存的旧容器；不要删除权重或共享缓存。更换镜像时重新验证patch适用性，而不是只改tag。
