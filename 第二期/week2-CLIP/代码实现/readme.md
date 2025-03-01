## 如何启动分布式训练

单机多卡
```bash
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0  train.py
```
node_rank 表示当前节点在所有节点中的排名，从 0 开始，一台机器一个节点

多机多卡
```bash
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="localhost" --master_port=12345 train.py
```
node_rank 表示当前节点在所有节点中的排名，从 0 开始，多台机器每个机器一个节点
master_addr 表示主节点的地址，多台机器时，主节点地址为 0 这台机器的地址
master_port 表示主节点的端口，多台机器时，主节点端口为 12345





