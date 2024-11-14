## 查找进程

```bash
ps aux | grep execution.sh
```

## 后台运行进程
- 默认 2 hour 后关闭

```
nohup /mnt/nlp/dh_ai/shell/execution.sh > script_output.log 2>&1 &
```

- 立刻关闭


```
nohup /mnt/nlp/dh_ai/shell/execution.sh --delay 0 > script_output.log 2>&1 &
```