import h5py
import yaml
import logging
import argparse
from pathlib import Path
from chesspos.preprocessing import (
	PgnExtractor, get_game_processor, get_game_filter
)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Extract information from games in pgn files.')
	parser.add_argument('--config', type=str, action="store", default="config.yaml",
		help='configuration file'
	)
	args = parser.parse_args()
	params = yaml.safe_load(open(Path(__file__).with_name(args.config)))['preprocess']

	position_extractor = PgnExtractor(
		pgn_path=params['pgn_path'],
		save_path=params['save_path'],
		game_processor=get_game_processor(params['game_processor']),
		game_filter=get_game_filter(params['game_filter']),
		chunk_size=params['chunk_size'],
		log_level=logging.INFO
	)
	position_extractor.extract(number_games=params['number_games'])
