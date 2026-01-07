import torch
from game_proccessing_utils import filter_by_score, map_all_to_next_move, rotate_all_games_to_all_directions, truncate_all_at_last_fruit
import gameserializer

games = gameserializer.read_games()

games = filter_by_score(games, min_score=15)
games = truncate_all_at_last_fruit(games)
# games = rotate_all_games_to_all_directions(games)

mapped_games = map_all_to_next_move(games)

print(f"Mapped Games: {len(mapped_games)}")
print(f"Total frames: {len([turn for game in mapped_games for turn in game])}")

DIRECTION_LABEL_MAP = {
	"UP": 0,
	"RIGHT": 1,
	"DOWN": 2,
	"LEFT": 3
}

REVERSE_DIRECTION_LABEL_MAP = {v: k for k, v in DIRECTION_LABEL_MAP.items()}

OPPOSITE_DIRECTION_MAP = {
	DIRECTION_LABEL_MAP["UP"]: DIRECTION_LABEL_MAP["DOWN"],
	DIRECTION_LABEL_MAP["DOWN"]: DIRECTION_LABEL_MAP["UP"],
	DIRECTION_LABEL_MAP["RIGHT"]: DIRECTION_LABEL_MAP["LEFT"],
	DIRECTION_LABEL_MAP["LEFT"]: DIRECTION_LABEL_MAP["RIGHT"]
}

def conv_direction_to_y_tensor(direction: str, curr_direction: str) -> list[int]:
	directions = [0, 0, 0, 0]

	directions[DIRECTION_LABEL_MAP[direction]] = 1

	# opposite direction is invalid, so use 0 to bias against

	# directions[OPPOSITE_DIRECTION_MAP[DIRECTION_LABEL_MAP[curr_direction]]] = 0

	return directions

x_tensor = torch.tensor([
	turn.head_board+turn.body_board+turn.fruit_board for game in mapped_games for turn in game
], dtype=torch.float32)

y_tensor = torch.tensor([
	conv_direction_to_y_tensor(turn.next_move, turn.curr_direction) for game in mapped_games for turn in game
], dtype=torch.float32)