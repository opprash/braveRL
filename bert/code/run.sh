bash

python3 run_classifier.py \

 --task_name=sim \

 --do_train=true \

 --do_eval=true \

 --data_dir=/home/opprash/Desktop/project/bert_project/data \

 --vocab_file=$/home/opprash/Desktop/project/bert_project/chinese_L-12_H-768_A-12/vocab.txt \

 --bert_config_file=/home/opprash/Desktop/project/bert_project/chinese_L-12_H-768_A-12/bert_config.json \

 --init_checkpoint=/home/opprash/Desktop/project/bert_project/chinese_L-12_H-768_A-12/bert_model.ckpt \

 --max_seq_length=128 \

 --train_batch_size=32 \

 --learning_rate=2e-5 \

 --num_train_epochs=3.0 \

 --output_dir=/home/opprash/Desktop/project/bert_project/mytask_output
