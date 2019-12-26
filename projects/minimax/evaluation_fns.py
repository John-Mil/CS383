from game import State


def open_cells(game, state):
    """
    Estimates utility from the number of open cells next to the current player.
    More open cells is preferable to fewer.
    """
    if game.is_terminal(state):
        return game.utility(state)
    else:
        # the number of open cells next to the current player
        open_cells = {cell for (cell, _) in game.get_actions(state)}

        # there are at best 8 open cells, so score is near 1 when things are
        # "good" for the current player, and near 0 when things are "bad"
        score = len(open_cells) / 8
        return -score if state.min_to_play else score


def open_cells_diff(game, state, scale=2):
    """
    Prioritizes moves in which the number of open cells for the current player is greater than the number of open cells
    for its opponent with a scale factor.
    """
    if game.is_terminal(state):
        return game.utility(state)
    else:
        curr_open_cells = len({cell for (cell, _) in game.get_actions(state)})
        new_state = State(
            board=state.board,
            min_pos=state.min_pos,
            max_pos=state.max_pos,
            min_to_play=not state.min_to_play
        )
        opp_open_cells = len({cell for (cell, _) in game.get_actions(new_state)})
        score = (curr_open_cells - scale * opp_open_cells) / 8

        return -score if state.min_to_play else score
