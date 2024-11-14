#!/bin/bash

hour=4
time=$(expr $hour \* 3600)

# 解析命令行参数
while [ "$#" -gt 0 ]; do
    if [ "$1" == "--delay" ]; then
        shift
        if [ "$1" ]; then
            time=$1
            shift
        else
            echo "使用了 --delay 选项但未提供值，将使用默认值2"
        fi
    else
        shift
    fi
done

echo "will sleep $time"
sleep $time
echo "wake up"

# 执行指定的代码
export $(cat /proc/1/environ | tr '\0' '\n' | grep MATCLOUD_CANCELTOKEN) && /mnt/nlp/dh_ai/shell/matncli node cancel -url https://matpool.com/api/public/node
echo "shutdown machine"
