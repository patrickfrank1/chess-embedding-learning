from typing import Tuple
import numpy as np
import chess

from chesspos.preprocessing.position_processors import board_to_tensor, tensor_to_board

def sample_generator_to_autoencoder_input(samples: np.ndarray) -> Tuple[np.ndarray]:
	"""Takes a sample generator output and returns a pair of input tensors for the autoencoder."""
	return tuple([_add_tensor_padding_dimension(samples), _add_tensor_padding_dimension(samples)])

def _add_tensor_padding_dimension(tensor: np.ndarray) -> np.ndarray:
	"""Add another dimension to a tensor to allow for 3D convolution layers."""
	return np.expand_dims(tensor, axis=-1)

def _add_batch_dimension(tensor: np.ndarray) -> np.ndarray:
	"""Add batch dimension to a tensor to emulate batch size 1 for a single sample."""
	return np.expand_dims(tensor, axis=0)

def board_to_tensor_with_padding(board: chess.Board) -> np.ndarray:
	"""
	Take a chess board and return a tensor representation of the board padded by a 
	batch dimension and an extra dimension for 3D convolutions.
	"""
	return _add_batch_dimension(_add_tensor_padding_dimension(board_to_tensor(board)))

def tensor_to_board_with_padding(tensor: np.ndarray) -> chess.Board:
	"""
	Take a tensor representation of a board with a batch dimension and an additional
	dimension for 3D convolutions and return a chess board.
	"""
	return tensor_to_board(tensor[0,:,:,:,0])
