#!/bin/sh
GPU=$1
DATA=$2
ARCH=$3
CKPT=$4

if [ $# -ne 4 ]
    then
        echo "Arguments error: <GPU_ID> <DATASET> <ARCH> <CKPT>"
        exit 1
    fi

python train.py \
  --eval \
  --resume $CKPT \
	--sgpu $GPU \
	-d $DATA \
	-a $ARCH \
	-n 64 \
	-m 0 \
	--name test
