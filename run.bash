python train_2d.py --repeated 5 --model_name PositionalEncoding
python train_2d.py --repeated 5 --model_name FourierEncoding
python train_2d.py --repeated 5 --model_name SinEncoding
python train_2d.py --repeated 5 --model_name FourierMLPEncoding
python train_2d.py --repeated 5 --model_name LSTMBasedEncoder
python train_2d.py --repeated 5 --model_name CNNBasedEncoder
python train_2d.py --repeated 5 --model_name TransformerEncoder
python train_2d.py --repeated 5 --model_name PositionalEncodingnew

python train_2d.py --repeated 1 --model_name FourierEncoding_IM --d_model 256 --batch_size 8192 --lr 0.0005
python train_2d.py --repeated 3 --model_name TransformerEncoder --d_model 32
python train_2d.py --repeated 3 --model_name FourierMLPEncoding --d_model 32

python train_2d.py --repeated 1 --model_name FourierEncoding_IM --d_model 2048 --batch_size 8192 --lr 0.0005 --device 1
