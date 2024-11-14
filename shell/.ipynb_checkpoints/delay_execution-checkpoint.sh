#!/bin/bash

hour=4
time=$(expr $hour \* 3600)
echo "sleep $time"
sleep 10

# 执行指定的代码
export $(cat /proc/1/environ | tr '\0' '\n' | grep MATCLOUD_CANCELTOKEN) && /mnt/nlp/dh_ai/shell/matncli node cancel -url https://matpool.com/api/public/node
