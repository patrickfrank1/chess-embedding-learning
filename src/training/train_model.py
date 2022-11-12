import argparse
from typing import Tuple
import chess
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from train.preprocessing import (
	sample_generator_to_autoencoder_input, board_to_tensor_with_padding,
	tensor_to_board_with_padding
)

from chesspos.preprocessing import SampleGenerator
from chesspos.models import ResnetAutoencoder, DenseAutoencoder, CnnAutoencoder

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Train a chess position embedding model.')
	parser.add_argument('--config', type=str, action="store", default="config.yaml",
		help='configuration file'
	)
	args = parser.parse_args()
	train_params = yaml.safe_load(open(Path(__file__).with_name(args.config)))['train']
	evaluate_params = yaml.safe_load(open(Path(__file__).with_name(args.config)))['evaluate']

	"""
	Optionally define a custom function here which formats samples from the
	sample generator and returns inputs to the neural network.
	"""
	def custom_sample_preprocessor(samples: np.ndarray) -> Tuple[np.ndarray]:
		raise NotImplementedError

	"""
	Optionally define a custom function here, which converts a neural network
	output to chess board.
	"""
	def custom_output_to_board(output: np.ndarray) -> chess.Board:
		raise NotImplementedError

	"""
	Optionally define a custom function here, which converts a chess board to
	a neural network input.
	"""
	def custom_board_to_input(board: chess.Board) -> np.ndarray:
		raise NotImplementedError

	# Adapt these if you implemented a custom function above.
	sample_preprocessor = sample_generator_to_autoencoder_input # or choose custom_sample_preprocessor
	output_to_board = tensor_to_board_with_padding # or choose custom_output_to_board
	board_to_input = board_to_tensor_with_padding # or choose custom_board_to_input

	data_params = train_params['data']
	model_params = train_params['model']
	evaluate_data_params = evaluate_params['data']
	evaluate_eval_params = evaluate_params['eval']

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
	#autoencoder.load()

	number_examples = evaluate_eval_params['number_examples']

	autoencoder.plot_best_samples(number_samples=number_examples)
	autoencoder.plot_worst_samples(number_samples=number_examples)

	board1 = chess.Board()
	board2 = chess.Board("rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/5NP1/PP2PPBP/RNBQ1RK1 b - - 0 6")
	sample1 = board_to_tensor_with_padding(board1)
	sample2 = board_to_tensor_with_padding(board2)
	autoencoder.interpolate(sample1=sample1, sample2=sample2)
