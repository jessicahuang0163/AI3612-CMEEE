#!/bin/bash

CBLUE_ROOT=../data/CBLUEDatasets
  
MODEL_TYPE=bert
MODEL_PATH=../bert-base-chinese
SEED=2022
LABEL_NAMES=(labels)

for TASK_ID in 1  
do
  echo ${TASK_ID}
  case ${TASK_ID} in
  0)
    HEAD_TYPE=linear
    BS=4
    EVALBS=16
    ;;
  1)
    HEAD_TYPE=linear_nested
    LABEL_NAMES=(labels labels2)
    BS=4
    EVALBS=16
    ;;
  2)
    HEAD_TYPE=crf
    BS=4
    EVALBS=16
    ;;
  3)
    HEAD_TYPE=crf_nested
    LABEL_NAMES=(labels labels2)
    ;;
  *)
    echo "Error ${TASK_ID}"
    exit -1
    ;;
  esac

  OUTPUT_DIR=../ckpts/${MODEL_TYPE}_${HEAD_TYPE}_${SEED}

  PYTHONPATH=../.. \
  python run_cmeee.py \
    --output_dir                  ${OUTPUT_DIR} \
    --report_to                   none \
    --overwrite_output_dir        true \
    \
    --do_train                    true \
    --do_eval                     true \
    --do_predict                  true \
    \
    --dataloader_pin_memory       False \
    --per_device_train_batch_size ${BS} \
    --per_device_eval_batch_size  ${EVALBS} \
    --gradient_accumulation_steps 4 \
    --eval_accumulation_steps     500 \
    \
    --learning_rate               9e-6 \
    --weight_decay                9e-6 \
    --max_grad_norm               0.35 \
    --lr_scheduler_type           cosine \
    \
    --num_train_epochs            20 \
    --warmup_ratio                0.05 \
    --logging_dir                 ${OUTPUT_DIR} \
    \
    --logging_strategy            steps \
    --logging_first_step          true \
    --logging_steps               200 \
    --save_strategy               steps \
    --save_steps                  1000 \
    --evaluation_strategy         steps \
    --eval_steps                  1000 \
    \
    --save_total_limit            1 \
    --no_cuda                     false \
    --seed                        ${SEED} \
    --dataloader_num_workers      8 \
    --disable_tqdm                true \
    --load_best_model_at_end      true \
    --metric_for_best_model       f1 \
    --greater_is_better           true \
    \
    --model_type                  ${MODEL_TYPE} \
    --model_path                  ${MODEL_PATH} \
    --head_type                   ${HEAD_TYPE} \
    \
    --cblue_root                  ${CBLUE_ROOT} \
    --max_length                  512 \
    --label_names                 ${LABEL_NAMES[@]} 
done