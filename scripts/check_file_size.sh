#!/bin/bash

max_size=10485760  # 10MB
oversized=false

for file in "$@"; do
  if [ -f "$file" ]; then
    size=$(stat -c%s "$file")
    if [ $size -gt $max_size ]; then
      echo "Error: 文件 '$file' 的大小为 $(($size / 1024 / 1024))MB，超过了 10MB 的限制。"
      oversized=true
    fi
  fi
done

if [ "$oversized" = true ]; then
  echo "提交被阻止。请移除或减小超大文件的体积后再尝试提交。"
  exit 1
fi

exit 0
