import chess
import yaml
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow import keras

from chesspos.preprocessing import SampleGenerator, board_to_tensor, tensor_to_board
from chesspos.models import ResnetAutoencoder

data_params = yaml.safe_load(open(Path(__file__).with_name('params_tensor.yaml')))['train']['data']
model_params = yaml.safe_load(open(Path(__file__).with_name('params_tensor.yaml')))['train']['model']

start_board = chess.Board()
start_tensor = board_to_tensor(start_board)
print(start_tensor)
slice0 = start_tensor[:,:,0]

sample_preprocessor = lambda samples: tuple([samples.reshape((*samples.shape, 1)), samples.reshape((*samples.shape, 1))])

train_generator = SampleGenerator(
	sample_dir = data_params['train_dir'],
	sample_preprocessor = sample_preprocessor,
	batch_size = data_params['train_batch_size'],
	sample_type = np.float32,
)

test_generator = SampleGenerator(
	sample_dir = data_params['test_dir'],
	sample_preprocessor = sample_preprocessor,
	batch_size = data_params['test_batch_size'],
	sample_type = np.float32,
)

generator_test = test_generator.get_generator()
sample = next(generator_test)
#print(f"len(sample): {len(sample)}")
#print(f"sample[0].shape: {sample[0].shape}")
#print(f"sample[0].dtype: {sample[0].dtype}")
#board = tensor_to_board(sample[0][0,:,:,:,0])
#print(board.__str__())
#print(f"sample[0][0,:,:,:,0]: {sample[0][0,:,:,:,0]}")

autoencoder = ResnetAutoencoder(
	train_generator=train_generator,
	test_generator=test_generator,
	train_steps_per_epoch=model_params['train_steps_per_epoch'],
	test_steps_per_epoch=model_params['test_steps_per_epoch'],
	save_dir=model_params['save_dir'],
	output_to_board = tensor_to_board,
	board_to_input = board_to_tensor,
	loss = model_params['loss'],
	tf_callbacks = model_params['tf_callbacks'] + [
		keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.03, patience=5, verbose=1, mode='min', restore_best_weights=True),
		#keras.callbacks.TensorBoard(log_dir=model_params['save_dir'], histogram_freq=1, write_images=True, embeddings_freq=1)
	]
)

history = autoencoder.train()
autoencoder.save()

start_board = chess.Board()
prediction = autoencoder.predict_from_board([start_board])
prediction = prediction[0,:,:,:,0]
prediction = autoencoder.binarize_array(prediction)
print(prediction[:,:,0])
print(prediction[:,:,1])
print(tensor_to_board(prediction).__str__())
