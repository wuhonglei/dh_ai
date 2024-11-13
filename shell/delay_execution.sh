#!/bin/bash

# 延迟 2 小时 (7200 秒)
echo "sleep 7200"
sleep 7200

# 执行指定的代码
export $(cat /proc/1/environ | tr '\0' '\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node
