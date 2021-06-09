#!/bin/sh
GPU=$1
DATA=$2
ARCH=$3
N=$4

if [ $# -ne 4 ]
    then
        echo "Arguments error: <GPU_ID> <DATASET> <ARCH> <BATCH_SIZE>"
        exit 1
    fi

python train.py \
	--lr 0.1 \
	--decay 1e-4 \
	--epoch 200 \
	--lamda 0.0 \
	-m 0 \
	--omega 0.0 \
	--sgpu $GPU \
	-d $DATA \
	-a $ARCH \
	-n $N \
	--name baseline_$DATA_$ARCH
