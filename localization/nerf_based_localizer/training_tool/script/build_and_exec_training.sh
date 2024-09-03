#!/bin/bash
set -eux

ROOT_DIR=$(readlink -f $(dirname $0)/../)
TRAIN_RESULT_DIR=$(readlink -f $1)
DATASET_DIR=$(readlink -f $2)

cd ${ROOT_DIR}
cmake . -B build
cmake --build build --config RelWithDebInfo -j8

cd ${ROOT_DIR}/build
rm -rf ${TRAIN_RESULT_DIR}
mkdir ${TRAIN_RESULT_DIR}
cp ${ROOT_DIR}/config/train_config.yaml ${TRAIN_RESULT_DIR}/
cp ${DATASET_DIR}/cams_meta.tsv ${TRAIN_RESULT_DIR}/
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
./main train ${TRAIN_RESULT_DIR} ${DATASET_DIR}
