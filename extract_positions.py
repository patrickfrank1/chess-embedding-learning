from typing import Callable
import h5py
import yaml
import logging
import argparse
from pathlib import Path
import chess

from chesspos.preprocessing import *

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Extract information from games in pgn files.')
	parser.add_argument('--config', type=str, action="store", default="config.yaml",
		help='configuration file'
	)
	args = parser.parse_args()
	params = yaml.safe_load(open(Path(__file__).with_name(args.config)))['preprocess']

	game_filter = None
	if params['game_filter'] == 'custom':
		def custom_filter(header: chess.pgn.Headers):
			return filter_by_elo(header, white_elo_range=[2600,4000], black_elo_range=[2600,4000]) and \
				filter_by_time_control(header, time_range=[5,60])
		game_filter = custom_filter
	else:
		game_filter = get_game_filter(params['game_filter'])


	position_extractor = PgnExtractor(
		pgn_path=params['pgn_path'],
		save_path=params['save_path'],
		game_processor=get_game_processor(params['game_processor']),
		game_filter=game_filter,
		chunk_size=params['chunk_size'],
		log_level=logging.INFO
	)
	position_extractor.extract(number_games=params['number_games'])
