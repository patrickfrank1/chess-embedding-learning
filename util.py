import chess
import numpy as np
from typing import Callable
from chesspos.preprocessing import (
	positions_to_bitboard, positions_to_tensor, positions_to_tensor_triplets,
	filter_out_bullet_games, filter_by_elo, filter_by_time_control
)

def get_game_processor(game_processor_name: str) -> Callable[[chess.pgn.Game], np.ndarray]:
	if game_processor_name == "positions_to_tensor":
		return positions_to_tensor
	elif game_processor_name == "positions_to_bitboard":
		return positions_to_bitboard
	elif game_processor_name == "positions_to_tensor_triplets":
		return positions_to_tensor_triplets
	else:
		raise ValueError(f"Unknown game_processor_name: {game_processor_name}")

def get_game_filter(game_filter_name: str) -> Callable[[chess.pgn.Header], bool]:
	if game_filter_name == "filter_out_bullet_games":
		return filter_out_bullet_games
	elif game_filter_name == "only_rapid_master_games":
		return _only_rapid_master_games
	else:
		raise ValueError(f"Unknown game_filter_name: {game_filter_name}")

def _only_rapid_master_games(header: chess.pgn.Headers) -> bool:
	return filter_by_elo(header, white_elo_range=[2600,4000], black_elo_range=[2600,4000]) and \
		filter_by_time_control(header, time_range=[5,60])