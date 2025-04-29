#!/usr/bin/env bash
# =============================================================
#  ppltn_rate.sh  ──  CARD 모델로 분단위 유동인구(ppltn_rate20) 예측
# =============================================================

torch.autograd.detect_anomaly()

# ---------- 로그 폴더 ----------
if [ ! -d "./logs/ShortForecasting" ]; then
    mkdir -p ./logs/ShortForecasting
fi

# ---------- (선택) Weights & Biases ----------
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="2ad137270048215c8e5b56aa238c2ee91d3afc06"
export WANDB_MODE=offline            # online 으로 바꾸면 자동 업로드

# ---------- 데이터 경로 ----------
ROOT=./dataset/KT/live_ppltn_stts    # 끝에 / 없음
DATA=live_ppltn_stts_preprocessed_real.csv       # CSV 파일명

# ---------- 실험 파라미터 ----------
model_name=CARD

# 예측 horizon 4가지 예시 (10·20·30·60분 뒤) ─ 한 GPU 당 하나씩 병렬
pred_lens=(10 20 30 60)             
cuda_ids=(0 1 2 3)                   # 사용 가능한 GPU 번호

seq_len=40     # 입력 길이
label_len=20   # 디코더 warm‑up 길이

# ---------- 루프 ----------
for ((i = 0; i < ${#pred_lens[@]}; i++)); do

    pred_len=${pred_lens[i]}
    export CUDA_VISIBLE_DEVICES=${cuda_ids[i]}
    
    python -u run.py \
    --num_workers 0 \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "ppltn_${seq_len}_${pred_len}" \
    --model ${model_name} \
    --data live_ppltn \
    --features M \
    --seq_len ${seq_len} \
    --label_len ${label_len} \
    --pred_len ${pred_len} \
    --freq t \
    --factor 3 \
    --enc_in 48  --dec_in 48  --c_out 48 \
    --e_layers 2  --d_layers 1 \
    --d_model 128 --n_heads 8 --d_ff 256 \
    --dropout 0.1 --fc_dropout 0.1 --head_dropout 0.0 \
    --patch_len 16 --stride 8 \
    --train_epochs 1000 --patience 10 \
    --batch_size 8 --learning_rate 0.002 \
    --des "Exp" --itr 1 \
    2>&1 | tee "logs/ShortForecasting/${model_name}_ppltn_${seq_len}_${pred_len}.log" &
done

wait   # 모든 백그라운드 프로세스 종료 대기
echo "==== ALL CARD RUNS FINISHED ===="