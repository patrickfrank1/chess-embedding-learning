import argparse
import logging
from pathlib import Path

import chess
import numpy as np
import yaml
from chesspos.preprocessing import PgnExtractor

import src.utils.util as util

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Extract information from games in pgn files.')
	parser.add_argument('--config', type=str, action="store", default="",
		help='configuration file'
	)
	args = parser.parse_args()

	config_path = Path(__file__).with_name('config.yaml')
	if args.config != "":
		config_path = Path(args.config)
	params = yaml.safe_load(open(config_path))

	"""
	Optionally define a custom filter function here, which filters game by the
	provided header information. Skips a game if the filter evaluates to True.
	"""
	def custom_game_filter(header: chess.pgn.Headers) -> bool:
		return False

	"""
	Optionally define a custom function here which encodes game positions for
	later use in a neural network
	"""
	def custom_game_processor(game: chess.pgn.Game) -> np.ndarray:
		raise NotImplementedError

	game_filter = None
	game_processor = None


	if params['game_filter'] is not None:
		game_filter = util.get_game_filter(params['game_filter'])
	else:
		game_filter = custom_game_filter

	if params['game_processor'] is not None:
		game_processor = util.get_game_processor(params['game_processor'])
	else:
		game_processor = custom_game_processor


	"""
	Instantiate PngExtractor
	"""
	position_extractor = PgnExtractor(
		pgn_path=params['pgn_path'],
		save_path=params['save_path'],
		game_processor=game_processor,
		game_filter=game_filter,
		chunk_size=params['chunk_size'],
		log_level=logging.INFO
	)

	"""
	Extract positions from pgn games.
	"""
	position_extractor.extract(number_games=int(params['number_games']))
