## 查找进程

```bash
ps aux | grep delay_execution.sh
```

## 后台运行进程

```
nohup ./shell/delay_execution.sh > script_output.log 2>&1 &
```