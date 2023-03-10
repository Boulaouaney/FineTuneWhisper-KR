python3 -m torch.distributed.launch \
	--nproc_per_node 6 --node_rank 1 FineTuningScript.py \
	--model_name_or_path="openai/whisper-large-v2" \
	--dataset_name="Bingsu/zeroth-korean" \
	--language="korean" \
	--train_split_name="train" \
	--eval_split_name="test" \
	--max_steps="5000" \
	--output_dir="./whisper-large-v2-Ko" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="16" \
	--logging_steps="25" \
	--learning_rate="1e-5" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--eval_steps="1000" \
	--save_strategy="steps" \
	--save_steps="1000" \
	--generation_max_length="225" \
	--preprocessing_num_workers="32" \
	--report_to="tensorboard" \
	--metric_for_best_model="wer" \
	--greater_is_better="False" \
	--load_best_model_at_end \
	--gradient_checkpointing \
	--fp16="True" \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate="True" \
	--use_auth_token="True" \
	--push_to_hub="True"
