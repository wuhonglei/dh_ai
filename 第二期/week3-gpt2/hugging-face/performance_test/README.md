## python 环境

```
(myconda) root@qM2N1M:/mnt/nlp/dh_ai# command -v python
/root/miniconda3/envs/myconda/bin/python
```

```
Python 3.12.4 （myconda'） ~/miniconda3/envs/myconda/bin/python
```

## 分布式评估

```bash
torchrun --nproc_per_node=4 eval.py
```

## 单机评估

```bash
python eval.py
```