# export CUDA_VISIBLE_DEVICES=7,6,5,4

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=720
model_name=CARD



root_path_name=./dataset/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=custom



random_seed=2021
for pred_len in 96 192 336 720

do
python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
    --e_layers 2 \
    --n_heads 2 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10 \
    --itr 1 --batch_size 128 --learning_rate 0.0001 --merge_size 2 \
    --lradj CARD --warmup_epochs 10 \
  2>&1 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log \

  

done
