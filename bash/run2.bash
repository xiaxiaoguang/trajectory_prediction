nohup python run.py --pre_model_name erpp --embed_name teaser --task_epoch 20 --device 1 &
nohup python run.py --pre_model_name gru --embed_name teaser --task_epoch 20 --device 0 &
nohup python run.py --pre_model_name rnn --embed_name teaser --task_epoch 20 --device 3 &
python run.py --pre_model_name stlstm --embed_name teaser --task_epoch 20 --device 2

python run.py --pre_model_name decoder --embed_name fourier --task_epoch 20 --embed_epoch 10 --device 0
python run.py --pre_model_name rnn --embed_name ctle --task_epoch 20 --embed_epoch 10 --device 0