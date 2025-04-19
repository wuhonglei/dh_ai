## 运行

分布式运行
```bash
torchrun --nproc_per_node=4 train.py
```

分布式运行后台运行
```bash
nohup torchrun --nproc_per_node=4 train.py > train.log 2>&1 &
```

查看日志
```bash
tail -f train.log
```

杀掉 train.py 进程
```bash
kill -9 $(pgrep -f train.py)
```

