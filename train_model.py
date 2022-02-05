import argparse
import chess
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from chesspos.preprocessing import SampleGenerator, board_to_tensor, tensor_to_board
from chesspos.models import ResnetAutoencoder, DenseAutoencoder, CnnAutoencoder

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Train a chess position embedding model.')
	parser.add_argument('--config', type=str, action="store", default="config.yaml",
		help='configuration file'
	)
	args = parser.parse_args()
	params = yaml.safe_load(open(Path(__file__).with_name(args.config)))['train']
	evaluate_params = yaml.safe_load(open(Path(__file__).with_name(args.config)))['evaluate']

	data_params = params['data']
	model_params = params['model']
	evaluate_data_params = evaluate_params['data']
	evaluate_eval_params = evaluate_params['eval']

	# Training script
	add_tensor_dimension = lambda x: np.expand_dims(x, axis=-1)
	add_batch_dimension = lambda x: np.expand_dims(x, axis=0)
	sample_preprocessor = lambda samples: tuple([add_tensor_dimension(samples), add_tensor_dimension(samples)])
	board_to_tensor_with_padding = lambda board: add_batch_dimension(add_tensor_dimension(board_to_tensor(board)))
	tensor_to_board_with_padding = lambda tensor: tensor_to_board(tensor[0,:,:,:,0])


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

	autoencoder = DenseAutoencoder(
		train_generator=train_generator,
		test_generator=test_generator,
		train_steps_per_epoch=model_params['train_steps_per_epoch'],
		test_steps_per_epoch=model_params['test_steps_per_epoch'],
		save_dir=model_params['save_dir'],
		output_to_board = tensor_to_board_with_padding,
		board_to_input = board_to_tensor_with_padding,
		loss = model_params['loss'],
		tf_callbacks = model_params['tf_callbacks'] + [
			keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.005, patience=10, verbose=1, mode='min', restore_best_weights=True),
			keras.callbacks.TensorBoard(log_dir=model_params['save_dir'], histogram_freq=1, write_images=True, embeddings_freq=1)
		]
	)

	history = autoencoder.train()
	autoencoder.save()
	autoencoder.load()

	number_examples = evaluate_eval_params['number_examples']

	autoencoder.plot_best_samples(number_samples=number_examples)
	autoencoder.plot_worst_samples(number_samples=number_examples)

	board1 = chess.Board()
	board2 = chess.Board("rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/5NP1/PP2PPBP/RNBQ1RK1 b - - 0 6")
	sample1 = board_to_tensor_with_padding(board1)
	sample2 = board_to_tensor_with_padding(board2)
	autoencoder.interpolate(sample1=sample1, sample2=sample2)