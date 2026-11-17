for i in 1 2 3 4 5 6; do /home/cgh/anaconda3/envs/sglang/bin/python scripts/upload_hf.py $1 && break; echo "attempt $i failed"; sleep 15; done
