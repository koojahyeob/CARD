#!/usr/bin/env bash
mkdir -p logs/LongForecasting

model_name=CARD
pred_lens=(96 192 336 720)
cuda_ids1=(0 1 2 3)

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY= 2ad137270048215c8e5b56aa238c2ee91d3afc06
export WANDB_MODE=offline

for i in "${!pred_lens[@]}"; do
  pred_len=${pred_lens[$i]}
  export CUDA_VISIBLE_DEVICES=${cuda_ids1[$i]}

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id "traffic_96_${pred_len}" \
    --model "$model_name" \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len "$pred_len" \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 --dec_in 862 --c_out 862 \
    --top_k 5 \
    --des 'Exp' --itr 1 \
    --e_layers 2 --n_heads 16 --d_model 128 --d_ff 256 \
    --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 \
    --patch_len 16 --stride 8 \
    --patience 100 --train_epochs 100 --lradj CARD \
    --batch_size 12 --learning_rate 0.0001 \
    --dp_rank 8 --warmup_epochs 0 \
    2>&1 | tee "logs/LongForecasting/${model_name}_traffic_96_${pred_len}.log" &
done

wait   # 모든 백그라운드 작업 종료 대기