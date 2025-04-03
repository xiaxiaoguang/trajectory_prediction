python run.py --pre_model_name erpp --task_epoch 20 --embed_epoch 10 --device 1
python run.py --pre_model_name stlstm --task_epoch 20 --embed_epoch 10 --device 2
python run.py --pre_model_name gru --task_epoch 20 --embed_epoch 10 --device 3
python run.py --pre_model_name transformer --task_epoch 20 --embed_epoch 10 --device 2 --hidden_size 128
python run.py --pre_model_name decoder --task_epoch 20 --embed_epoch 10 --device 3 --hidden_size 128

python run.py --embed_name fourier --pre_model_name rnn --task_epoch 20 --embed_epoch 5 --device 3 --hidden_size 128
python run.py --embed_name fourier --pre_model_name decoder --task_epoch 20 --embed_epoch 5 --device 3 --hidden_size 128
python run.py --embed_name fourier --pre_model_name transformer --task_epoch 20 --embed_epoch 5 --device 3 --hidden_size 128
python run.py --embed_name fourier --pre_model_name erpp --task_epoch 20 --embed_epoch 5 --device 1 --hidden_size 128
python run.py --embed_name fourier --pre_model_name stlstm --task_epoch 20 --embed_epoch 5 --device 2 --hidden_size 128
python run.py --embed_name fourier --pre_model_name gru --task_epoch 20 --embed_epoch 5 --device 3 --hidden_size 128


python run.py --pre_model_name erpp --embed_name hier --task_epoch 20 --device 1
python run.py --pre_model_name stlstm --embed_name hier --task_epoch 20 --device 2
python run.py --pre_model_name rnn --embed_name hier --task_epoch 20 --device 3
python run.py --pre_model_name gru --embed_name hier --task_epoch 20 --device 1
python run.py --pre_model_name transformer  --embed_name hier --task_epoch 20  --device 2
python run.py --pre_model_name decoder  --embed_name hier --task_epoch 20 --device 0
