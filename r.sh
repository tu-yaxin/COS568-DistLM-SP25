cd /proj/cos568proj2-PG0/groups/yt9801/COS568-DistLM-SP25/
export GLUE_DIR=/proj/cos568proj2-PG0/glue_data
export TASK_NAME=RTE
# rm -r /tmp/2a/
mkdir -p /tmp/2a/
mkdir -p /tmp/2a/cache/ # you have to initialize a new cache folder each time or you are using outdated cache and your model crashes!!!!!!

python3 run_glue2a.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir /tmp/2a/ \
  --cache_dir /tmp/2a/cache/ \
  --local_rank ${1} \
  --overwrite_output_dir \
  --master_ip 10.10.1.2 \
  --master_port 51296 \
  --world_size=4

