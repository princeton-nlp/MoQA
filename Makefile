data-config:
	$(eval DATA_DIR=data)
	$(eval SAVE_DIR=output)
	$(eval CACHE_DIR=cache)

test-gpt3: data-config
	python test_gpt.py \
		--train_data $(DATA_DIR)/train.json \
		--test_data $(DATA_DIR)/test.json  \
		--question_types short,medium,long,yesno \
		--ex_per_class 2 \
		--num_trials 3 

run-reader: model-name reader-data nq-open-data-all reader-params
	python run_reader.py \
		--model_name_or_path roberta-base \
		--cache_dir $(CACHE_DIR) \
		--do_train True \
		--do_eval True \
		--do_predict False \
		--optim adamw_torch \
		--per_device_train_batch_size 4 \
		--per_device_eval_batch_size 4 \
		--gradient_accumulation_steps 1 \
		--learning_rate 2e-5 \
		--num_train_epochs 10 \
		--max_seq_length 300 \
		--question_type short,medium,long,yesno \
		--train_file $(DATA_DIR)/train.json \
		--validation_file $(DATA_DIR)/dev.json \
		--test_file $(DATA_DIR)/test.json \
		--train_pred_file $(DATA_DIR)/train_dpr-all_pred.json \
		--validation_pred_file $(DATA_DIR)/dev_dpr-all_pred.json \
		--test_pred_file $(DATA_DIR)/test_dpr-all_pred.json \
		--preprocessing_num_workers 32 \
		--num_train_passages 24 \
		--num_eval_passages 50 \
		--num_answers 10 \
		--evaluation_strategy "epoch" \
		--save_strategy "epoch" \
		--load_best_model_at_end True \
		--metric_for_best_model eval_f1 \
		--greater_is_better True \
		--report_to "all" \
		--save_total_limit 2 \
		--n_best_size 1 \
		--ddp_find_unused_parameters False \
		--max_answer_length 300 \
		--output_dir $(SAVE_DIR)/reader-all

run-classifier: 
	python run_classifier.py \
		--model_name_or_path roberta-base \
		--cache_dir $(CACHE_DIR) \
		--do_train \
		--do_eval \
		--do_predict \
		--optim adamw_torch \
		--per_device_train_batch_size 8 \
		--per_device_eval_batch_size 4 \
		--gradient_accumulation_steps 1 \
		--learning_rate 1e-5 \
		--num_train_epochs 10 \
		--max_seq_length 50 \
		--train_file $(DATA_DIR)/train.json \
		--validation_file $(DATA_DIR)/dev.json \
		--test_file $(DATA_DIR)/test.json \
		--evaluation_strategy "epoch" \
		--save_strategy "steps" \
		--save_steps 2000 \
		--report_to "all" \
		--ddp_find_unused_parameters False \
		--save_total_limit 2 \
		--output_dir $(SAVE_DIR)/classifier