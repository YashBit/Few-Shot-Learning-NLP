export TASK_NAME="cola"
export out_dir="/scratch/ks4765/research/w21_robustness/mbe/model/test/0shot-cola"
export train_file="/scratch/ks4765/research/w21_robustness/mbe/data/glue/rte/dev.csv"
export val_file="/scratch/ks4765/research/w21_robustness/mbe/data/glue/rte/dev.csv"
export test_file="/scratch/ks4765/research/w21_robustness/mbe/data/cola-ood/cola_ood_private_test.csv"
export ckpt_path="/scratch/ks4765/research/w21_robustness/mbe/model/rob-base-mnli"


python3 train.py \
  --model_name_or_path roberta-base \
  --do_eval \
  --task_name $TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 6 \
  --logging_steps 100 \
  --evaluation_strategy epoch \
  --output_dir ${out_dir} \
  --overwrite_output_dir \
  --use_matthews True


 
  #   --train_file ${train_file} \
  # --validation_file ${val_file} \
  # --test_file ${test_file} \