#!/bin/bash

#### Analysis ####
analysis=$1

#### Epochs ####
start_epoch=0
epochs=1

#### Optimization ####
optimizer="adam"
learning_rate=1e-3 
momentum=0.99

#### Data Loader ####
batch_size=4
num_workers=2

python unet_train.py    --analysis $analysis \
                        --gpu \
                        --save \
                        --start-epoch $start_epoch \
                        --epochs $epochs \
                        --optimization $optimizer  \
                        --learning-rate $learning_rate \
                        --momentum $momentum \
                        --worker $num_workers \
                        --batch-size $batch_size \
