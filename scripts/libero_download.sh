#!/bin/bash
# Download the LIBERO-10 demonstration files (13.7 GB) with resume and retries. Usage: libero_download.sh <proxy port> <0|1> (odd/even split)
cd /data1/cgh/dataset/libero; export https_proxy=http://127.0.0.1:$1; i=0
while read p size; do
  i=$((i+1)); [ $((i%2)) = $2 ] || continue
  f=$(basename $p)
  for t in $(seq 1 200); do
    have=$(stat -c %s $f 2>/dev/null || echo 0); [ "$have" = "$size" ] && break
    curl -sS -L -C - --connect-timeout 15 --speed-time 30 --speed-limit 20000 -o $f "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/$p" || sleep 5
  done
  echo "done $f $(stat -c %s $f)"
done < files.txt
echo ALL DONE
