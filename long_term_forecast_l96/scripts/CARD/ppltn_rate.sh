#!/usr/bin/env bash
# =============================================================
#  ppltn_rate.sh  ──  CARD 모델로 분단위 유동인구(ppltn_rate20) 예측
# =============================================================
# & ".\scripts\CARD\ppltn_rate.sh"

torch.autograd.detect_anomaly()

# ---------- 로그 폴더 ----------
if [ ! -d "./logs/ShortForecasting" ]; then
    mkdir -p ./logs/ShortForecasting
fi

# ---------- (선택) Weights & Biases ----------
# export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_API_KEY=""
# export WANDB_MODE=offline            # online 으로 바꾸면 자동 업로드

#DATA=live_ppltn_stts_241101_250610_preprocessed_real.csv       # CSV 파일명
# data_length=241101_250610_run_250611_ver1  # 공백 제거 및 변수명 수정
# 실제로 241125부터 데이터 들어가고, 성수와 홍대입구역(2호선) 인구수는 제외함 (결측값 너무 많아서)

# ---------- 데이터 경로 ----------
# ROOT=./dataset/KT/live_ppltn_stts    # 끝에 / 없음
ROOT=./dataset/KT/live_ppltn    # 끝에 / 없음
# CSV 파일명
DATA=live_ppltn_num_241101_250610_preprocessed_real.csv 
# DATA=live_ppltn_stts_241101_250610_preprocessed_real_without_ppltn_rate40__GN.csv      # CSV 파일명
# data_length=250401_250507_run_250613_ver2  # 공백 제거 및 변수명 수정
data_length=241101_250610_run_250625_num # 공백 제거 및 변수명 수정
# ---------- 실험 파라미터 ----------7
model_name=CARD

# 예측 horizon 4가지 예시 (10·20·30·60분 뒤) ─ 한 GPU 당 하나씩 병렬
pred_lens=(48)   # 예측 길이 (시퀀스 단위 10 -> 50분)      
cuda_ids=(0)                   # 사용 가능한 GPU 번호

seq_len=96    # 입력 길이
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
    --enc_in 32  --dec_in 32  --c_out 32 \
    --e_layers 4  --d_layers 2 \
    --d_model 256 --n_heads 8 --d_ff 512 \
    --dropout 0.2 --fc_dropout 0.2 --head_dropout 0.0 \
    --patch_len 16 --stride 8 \
    --train_epochs 100 --patience 10 \
    --batch_size 32 --learning_rate 0.001 \
    --des "Exp" --itr 1 \
    --output_attention \
    --data_length "${data_length}" \
    2>&1 | tee "logs/ShortForecasting/${model_name}_ppltn_${seq_len}_${pred_len}_${data_length}.log" &
done

wait   # 모든 백그라운드 프로세스 종료 대기
echo "==== ALL CARD RUNS FINISHED ===="

# Attention 가중치 출력 (시각화 용도) --output_attention \