from typing import Callable

import numpy as np
import chess
import chess.pgn

import chesspos.preprocessing as cp
import chesspos.preprocessing.game_filters as gf
import chesspos.preprocessing.position_filters as pf
import chesspos.preprocessing.position_processors as pp
import chesspos.preprocessing.position_aggregators as pa
import chesspos.custom_types as ct

### Custom implementation of game filters

SUPPORTED_GAME_FILTERS = [
	"no_filter",
	"filter_out_bullet_games",
	"master_vs_master",
	"master_vs_amateur",
	"draw"
]
BULLET_GAME_THRESHOLD = 1 # one minute
TIME_THRESHOLD_MIN = 0
TIME_THRESHOLD_MAX = 1000
ELO_MASTER_THRESHOLD = 2600
ELO_AMATEUR_THRESHOLD = 2200
ELO_THRESHOLD_MIN = 0
ELO_THRESHOLD_MAX = 4000

def get_game_filter(game_filter_name: str) -> ct.GameFilter:
	if game_filter_name == "no_filter":
		return gf.no_filter
	elif game_filter_name == "filter_out_bullet_games":
		return gf.time_control_filter(time_range_minutes=[BULLET_GAME_THRESHOLD, TIME_THRESHOLD_MAX])
	elif game_filter_name == "master_vs_master":
		return gf.elo_filter(
			white_elo_range=[ELO_MASTER_THRESHOLD, ELO_THRESHOLD_MAX],
			black_elo_range=[ELO_MASTER_THRESHOLD, ELO_THRESHOLD_MAX]
		)
	elif game_filter_name == "master_vs_amateur":
		def filter(header: chess.pgn.Headers) -> bool:
			return gf.elo_filter(
				white_elo_range=[ELO_MASTER_THRESHOLD, ELO_THRESHOLD_MAX],
				black_elo_range=[ELO_THRESHOLD_MIN, ELO_AMATEUR_THRESHOLD]
			)(header) or gf.elo_filter(
				white_elo_range=[ELO_THRESHOLD_MIN, ELO_AMATEUR_THRESHOLD],
				black_elo_range=[ELO_MASTER_THRESHOLD, ELO_THRESHOLD_MAX]
			)(header)
		return filter
	elif game_filter_name == "draw":
		def filter(header: chess.pgn.Headers) -> bool:
			return not gf.white_wins(header) and not gf.black_wins(header)
		return filter
	else:
		raise ValueError(f"Unsupported is_process_game value: {game_filter_name}. Supported values are: {SUPPORTED_GAME_FILTERS}")

### Custom implementation of position filters

SUPPORTED_POSITION_FILTERS = [
	"no_filter",
	"3_to_5_pieces",
	"subsample_opening_light",
	"subsample_opening_medium",
	"subsample_opening_heavy"
]

def get_position_filter(position_filter_name: str) -> ct.PositionFilter:
	if position_filter_name == "no_filter":
		return pf.no_filter
	elif position_filter_name == "3_to_5_pieces":
		return pf.filter_piece_count(min_pieces=3, max_pieces=5)
	elif position_filter_name == "subsample_opening_light":
		return pf.subsample_opening_10_linear
	elif position_filter_name == "subsample_opening_medium":
		return pf.subsample_opening_20_linear
	elif position_filter_name == "subsample_opening_heavy":
		return pf.subsample_opening_10_quadratic
	else:
		raise ValueError(f"Unsupported is_process_position value: {position_filter_name}. Supported values are: {SUPPORTED_POSITION_FILTERS}")

### Custom implementation of position processors

SUPPORTED_GAME_PROCESSORS = [
	"bitboard",
	"tensor"
]

def get_game_processor(game_processor_name: str) -> Callable[[chess.pgn.Game], np.ndarray]:
	if game_processor_name == "tensor":
		return pp.board_to_tensor
	elif game_processor_name == "bitboard":
		return pp.board_to_bitboard
	else:
		raise ValueError(f"Unsupported game_processor value: {game_processor_name}. Supported values are: {SUPPORTED_GAME_PROCESSORS}")

### Custom implementations of position aggregators

SUPPORTED_POSITION_AGGREGATORS = [
	"identity",
	"triplets"
]

def get_position_aggregator(position_aggregator_name: str) -> Callable[[np.ndarray], np.ndarray]:
	if position_aggregator_name == "identity":
		return lambda x: x
	elif position_aggregator_name == "triplets":
		return pa.encodings_to_tensor_triplets
	else:
		raise ValueError(f"Unsupported position_aggregator value: {position_aggregator_name}. Supported values are: {SUPPORTED_POSITION_AGGREGATORS}")
