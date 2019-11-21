# 这里是bert项目相关的学习
## 详情可参考：
## [使用bert做文本分类](https://www.jiqizhixin.com/articles/2019-03-13-4)  
#### 最后的命令参数这边使用过脚本的形式，但是无法运行，可能是作者专用吧（滑稽）
同时也可以使用一下命令的形式：  
* 训练命令：python run_classifier.py --task_name=mytask --do_train=true --do_eval=true --data_dir=/braveRl/bert/data --vocab_file=/braveRl/bert/chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=/braveRl/bert/chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=/braveRl/bert/chinese_L-12_H-768_A-12/bert_model.ckpt  --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=/braveRl/bert/mytask_output
* 验证命令：python run_classifier.py --task_name=mytask --do_eval=true --data_dir=/braveRl/bert/data --vocab_file=/braveRl/bert/chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=/braveRl/bert/chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=/braveRl/bert/chinese_L-12_H-768_A-12/bert_model.ckpt  --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=/braveRl/bert/mytask_output
* 预测命令：python run_classifier.py --task_name=mytask --do_predict=true --data_dir=/braveRl/bert/data --vocab_file=/braveRl/bert/chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=/braveRl/bert/chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=/braveRl/bert/chinese_L-12_H-768_A-12/bert_model.ckpt  --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=/braveRl/bert/mytask_output
#### 注：注意路径和你自己安装的位置匹配，注意下载bert中文模型解压在chinese_L-12_H-768_A-12这个文件夹下面就行了
