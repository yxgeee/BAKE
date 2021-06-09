#!/bin/sh
GPU=$1
DATA=$2
ARCH=$3
N=$4
M=$5
OMEGA=$6

if [ $# -ne 6 ]
    then
        echo "Arguments error: <GPU_ID> <DATASET> <ARCH> <BATCH_SIZE> <M> <OMEGA>"
        exit 1
    fi

python train.py \
	--lr 0.1 \
	--decay 1e-4 \
	--epoch 200 \
	--lamda 1.0 \
	--temp 4.0 \
	--sgpu $GPU \
	-d $DATA \
	-a $ARCH \
	-n $N \
	-m $M \
	--omega $OMEGA \
	--name BAKE_$DATA_$ARCH
