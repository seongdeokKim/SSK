# SSK

python finetune_plm.py --model_fn ./utils/models/model.pth --pretrained_model_name bert-base-uncased --train_fn ./data/dataset_for_train.txt --test_fn ./data/dataset_for_test --gpu_id 0 --batch_size 8 --n_epochs 10 --lr 5e-5 --max_length 256
