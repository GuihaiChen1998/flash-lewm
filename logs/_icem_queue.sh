cd /data1/cgh/code/flash-lewm
xargs -d '\n' -P 13 -I CMD bash -c CMD < logs/_icem_jobs.sh
echo QUEUE DONE
