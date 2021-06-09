#!/usr/bin/env bash

set -x

PARTITION=$1
WORK_DIR=$2
CKP=$3
PY_ARGS=${@:4}

GPUS=${GPUS:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

srun --mpi=pmi2 -p ${PARTITION} \
    --job-name=test \
    --gres=gpu:${GPUS} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/test_net.py --cfg ${WORK_DIR}/config.yaml \
          TEST.WEIGHTS ${CKP} \
          LAUNCHER slurm \
          OUT_DIR ${WORK_DIR} \
          PORT ${PORT} \
          ${PY_ARGS}
