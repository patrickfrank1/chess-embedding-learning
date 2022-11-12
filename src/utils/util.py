from typing import Callable

import numpy as np
import chess

import chesspos.preprocessing as cp

def get_game_processor(game_processor_name: str) -> Callable[[chess.pgn.Game], np.ndarray]:
	if game_processor_name == "positions_to_tensor":
		return cp.positions_to_tensor
	elif game_processor_name == "positions_to_bitboard":
		return cp.positions_to_bitboard
	elif game_processor_name == "positions_to_tensor_triplets":
		return cp.positions_to_tensor_triplets
	else:
		raise ValueError(f"Unknown game_processor_name: {game_processor_name}")

def get_game_filter(game_filter_name: str) -> Callable[[chess.pgn.Headers], bool]:
	if game_filter_name == "filter_out_bullet_games":
		return cp.filter_out_bullet_games
	elif game_filter_name == "only_rapid_master_games":
		return _only_rapid_master_games
	elif game_filter_name == "":
		raise ValueError(f"Unknown game_filter_name: {game_filter_name}")

def _only_rapid_master_games(header: chess.pgn.Headers) -> bool:
	return cp.filter_by_elo(header, white_elo_range=[2600,4000], black_elo_range=[2600,4000]) and \
		cp.filter_by_time_control(header, time_range=[5,60])