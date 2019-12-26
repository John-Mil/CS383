# Module to play test playing isolation with alpha-beta pruning

import time

from game import Game
from random_agent import RandomAgent
from minimax_agent import MinimaxAgent
import evaluation_fns


def time_games(board_size, min_agent, max_agent):
    g = Game(board_size)
    t0 = time.time()
    _, _ = g.play(min_agent, max_agent, verbose=False)
    t_delta = time.time() - t0
    print(f'Time to play a single game: {t_delta}')


if __name__ == '__main__':

    #  Test alpha-beta pruning

    min_agent = MinimaxAgent(depth_limit=3, eval_fn=evaluation_fns.open_cells_diff)
    min_agent_prune = MinimaxAgent(prune=True, depth_limit=3, eval_fn=evaluation_fns.open_cells_diff)
    max_agent = RandomAgent()

    time_games(4, min_agent, max_agent)  # 27.7 seconds
    time_games(4, min_agent_prune, max_agent)  # 7.7 seconds
