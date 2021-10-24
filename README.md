# SSK (Social Science Korea)

@ below command is execution line for fine-tuning pretrained language model
python finetune.py --model_fn ./utils/models/bert_1.pth --pretrained_model_name bert-base-uncased --train_fn ./data/dataset_for_train.txt --test_fn ./data/dataset_for_test.txt --gpu_id 0 --batch_size 8 --n_epochs 10 --lr 5e-5 --max_length 256
