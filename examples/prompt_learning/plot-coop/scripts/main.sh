#!/bin/bash

cd ..

# custom config
DATA=../DATA
TRAINER=PLOT 

DATASET=$1
CFG=rn50  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
# SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
N=$2  # number of proxy


for SHOTS in #TODO: fill
do
for SEED in 1 2 3
do
for ktype in #TODO: fill
do
for khp in #TODO: fill
do
for lda in #TODO: fill
do
rm -r ./output/OP_N${N}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
DIR=./output/OP_N${N}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}

mkdir -p "./${DATASET}_${N}/${SHOTS}/${SEED}/${ktype}/${khp}"

echo "Run this job and save the output to ${DIR}"
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--lda ${lda} \
--ktype ${ktype} \
--khp ${khp} \
TRAINER.PLOT.N_CTX ${NCTX} \
TRAINER.PLOT.CSC ${CSC} \
TRAINER.PLOT.CLASS_TOKEN_POSITION ${CTP} \
DATASET.NUM_SHOTS ${SHOTS} \
TRAINER.PLOT.N ${N} \
> "./${DATASET}_${N}/${SHOTS}/${SEED}/${ktype}/${khp}/${lda}.txt"
done
done
done
done
done
