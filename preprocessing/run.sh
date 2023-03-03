#!/bin/bash

N_TRAIN_SHARDS=8
N_VALID_SHARDS=1

DATA_FOLDER="../data/pretrain"
SOURCE_FOLDER="${DATA_FOLDER}/bnc"
SHARD_FOLDER="${DATA_FOLDER}/shard"
TOKENIZED_FOLDER="${DATA_FOLDER}/tokenized"
TRAIN_FILE="${SOURCE_FOLDER}/train.md"
TOKENIZER="wordpiece"
VOCAB_FILE="${DATA_FOLDER}/${TOKENIZER}_vocab.json"

SHARD_JOB_ID=`sbatch shard.sh ${SOURCE_FOLDER} ${SHARD_FOLDER} ${N_TRAIN_SHARDS} ${N_VALID_SHARDS} | awk '{ print $4 }'`
echo "Sharding job ${SHARD_JOB_ID} submitted"
VOCAB_JOB_ID=`sbatch create_vocab.sh ${TRAIN_FILE} ${VOCAB_FILE} ${TOKENIZER} | awk '{ print $4 }'`
echo "Vocab job ${VOCAB_JOB_ID} submitted"

for (( i=0; i<${N_TRAIN_SHARDS}; i++ )); do
    CACHE_JOB_ID=`sbatch --job-name="PRE-${i}T_INPUT" --output="preprocess_train_input_${i}.out" --dependency=afterok:${SHARD_JOB_ID}:${VOCAB_JOB_ID} tokenize_input.sh "${SHARD_FOLDER}/train_${i}.md" "${VOCAB_FILE}" "${TOKENIZED_FOLDER}/train_${i}.pickle.gz" | awk '{ print $4 }'`
    echo "Train cache ${i} job ${CACHE_JOB_ID} submitted"
done

for (( i=0; i<${N_VALID_SHARDS}; i++ )); do
    CACHE_JOB_ID=`sbatch --job-name="PRE-${i}V_INPUT" --output="preprocess_valid_input_${i}.out" --dependency=afterok:${SHARD_JOB_ID}:${VOCAB_JOB_ID} tokenize_input.sh "${SHARD_FOLDER}/valid_${i}.md" "${VOCAB_FILE}" "${TOKENIZED_FOLDER}/valid_${i}.pickle.gz" | awk '{ print $4 }'`
    echo "Valid cache ${i} job ${CACHE_JOB_ID} submitted"
done
