from math import inf
from agent import Agent


class MinimaxAgent(Agent):
    depth_limit = None
    eval_fn = None
    prune = None

    def __init__(self, eval_fn=None, depth_limit=inf, prune=False):
        self.depth_limit = depth_limit
        self.eval_fn = eval_fn
        self.prune = prune

    def select_action(self, game, state):
        """
        Return minimax decision from the current state
        INPUT
        game: Game instance
        state: State instance
        OUTPUT
        action: Action instance with highest utility
        """

        def min_value(state, depth, alpha=None, beta=None):
            """Return minimum utility value of successor states of state"""
            if game.is_terminal(state):
                return game.utility(state)
            if depth >= self.depth_limit:
                return self.eval_fn(game, state)
            v = inf
            if self.prune:
                for action in game.get_actions(state):
                    v = min(v, max_value(game.apply_action(state, action), depth+1, alpha, beta))
                    if v <= alpha: return v
                    beta = min(beta, v)
                return v
            else:
                for action in game.get_actions(state):
                    v = min(v, max_value(game.apply_action(state, action), depth+1))
                return v

        def max_value(state, depth, alpha=None, beta=None):
            """Return maximum utility value of successor states of state"""
            if game.is_terminal(state):
                return game.utility(state)
            if depth >= self.depth_limit:
                return self.eval_fn(game, state)
            v = -inf
            if self.prune:
                for action in game.get_actions(state):
                    v = max(v, min_value(game.apply_action(state, action), depth+1, alpha, beta))
                    if v >= beta: return v
                    alpha = max(alpha, v)
                return v
            else:
                for action in game.get_actions(state):
                    v = max(v, min_value(game.apply_action(state, action), depth+1))
                return v

        if state.min_to_play:
            # Return action such that utility of Result(state, action) is minimum
            best_val = inf
            best_action = None
            alpha = -inf
            beta = inf
            for action in game.get_actions(state):
                if self.prune:
                    val = max_value(game.apply_action(state, action), 1, alpha, beta)
                else:
                    val = max_value(game.apply_action(state, action), 1)
                if val < best_val:
                    best_val = val
                    best_action = action
            return best_action
        else:
            # Return action such that utility of Result(state, action) is maximum
            best_val = -inf
            best_action = None
            alpha = -inf
            beta = inf
            for action in game.get_actions(state):
                if self.prune:
                    val = min_value(game.apply_action(state, action), 1, alpha, beta)
                else:
                    val = min_value(game.apply_action(state, action), 1)
                if val > best_val:
                    best_val = val
                    best_action = action
            return best_action
