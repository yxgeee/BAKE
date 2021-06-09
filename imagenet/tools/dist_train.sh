#!/usr/bin/env bash

set -x

PYTHON=${PYTHON:-"python"}

WORK_DIR=$1
CONFIG=$2
PY_ARGS=${@:3}

GPUS=${GPUS:-8}

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

mkdir -p $WORK_DIR

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
tools/train_net.py --cfg ${CONFIG} \
    LAUNCHER pytorch OUT_DIR ${WORK_DIR} PORT ${PORT} ${PY_ARGS} | tee ${WORK_DIR}/log.txt
