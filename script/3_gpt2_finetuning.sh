python3 ./transformers/examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path=rinna/japanese-gpt2-medium \
    --train_file=./txt/gpt2_train_data.txt \
    --validation_file=./txt/gpt2_train_data.txt \
    --do_train \
    --do_eval \
    --num_train_epochs=100 \
    --save_steps=10000 \
    --save_total_limit=3 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --output_dir=output/ \
    --use_fast_tokenizer=False \
    --overwrite_output_dir
