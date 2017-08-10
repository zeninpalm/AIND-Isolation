"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
from math import inf, sqrt

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    free_space_ratio = len(game.get_blank_spaces()) / (game.width * game.height)

    if free_space_ratio > 0.9:
        # We still in the start phase of the game, center location should be more advantageous
        return float(weighted_open_moves(game, player))
    elif free_space_ratio > 0.5:
        # In the early phase of the game, we should move sparse space
        return float(free_spaces_of_neighborhood(game, player) - free_spaces_of_neighborhood(game, game.get_opponent(player)))
    elif free_space_ratio > 0.2:
        # Now we are almost in the final phase, try to stay away from opponent
        return float(distance_score(game.get_player_location(player), game.get_player_location(game.get_opponent(player))))
    else:
        #Now the board is almost full, legal_moves is the most important indicator
        return float(len(game.get_legal_moves(player)))

def weighted_open_moves(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    all_moves = [(i, j) for i in range(0, 7) for j in range(0, 7)]
    weighted_board = {(row, col): assign_weight_to_move((row, col)) for (row, col) in all_moves}
    legal_moves = game.get_legal_moves(player)

    own_score = 0
    for legal_move in legal_moves:
        own_score += weighted_board[legal_move]

    return own_score


def assign_weight_to_move(move):
    base_score = 6
    base_dist = 3

    while base_dist >= 0:
        if abs(move[0] - 3) >= base_dist and abs(move[1] - 3) >= base_dist:
            return base_score
        elif abs(move[0] - 3) >= base_dist or abs(move[1] - 3) >= base_dist:
            return base_score - 1

        base_score -= 2
        base_dist -= 1


def free_spaces_of_neighborhood(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    blank_spaces = game.get_blank_spaces()
    location = game.get_player_location(player)
    neighbor_hood_blanks = [l for l in blank_spaces if abs(l[0] - location[0]) <= 2 or abs(l[1] - location[1]) <= 2]
    return len(neighbor_hood_blanks)


def distance_score(loc1, loc2):
    return sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2) + 1


def custom_score_random(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return random.randrange(-100, 100)

def custom_score_2(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return free_spaces_of_neighborhood(game, player) ** 2 - free_spaces_of_neighborhood(game, game.get_opponent(player))

def custom_score_3(game, player):
    return 1.


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=5, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move = self.minimax(game, self.search_depth)

        except SearchTimeout:
            if best_move == (-1, -1):
                legal_moves = game.get_legal_moves(self)

                if len(legal_moves) > 0:
                    best_move = game.get_legal_moves(self)[0]  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        if best_move == (-1, -1):
            legal_moves = game.get_legal_moves(self)

            if len(legal_moves) > 0:
                best_move = game.get_legal_moves(self)[0]
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return self._max_value(game, depth)[1]

    def _check_timeout(self):
        if self.time_left() < self.TIMER_THRESHOLD:
                    raise SearchTimeout()

    def _reaches_terminal(self, legal_moves, depth):
        self._check_timeout()
        return (len(legal_moves) <= 0) or (depth <= 0)

    def _min_value(self, game, depth):
        self._check_timeout()
        legal_moves = game.get_legal_moves(game.active_player)
        if self._reaches_terminal(game.get_legal_moves(game.active_player), depth):
            return self.score(game, self), None

        value = inf
        min_move = None
        for move in legal_moves:
            v, _ = self._max_value(game.forecast_move(move), depth - 1)
            if v < value:
                value, min_move = v, move

        return value, min_move

    def _max_value(self, game, depth):
        self._check_timeout()
        legal_moves = game.get_legal_moves(game.active_player)
        if self._reaches_terminal(game.get_legal_moves(game.active_player), depth):
            return self.score(game, self), None

        value = -inf
        max_move = None
        for move in legal_moves:
            v, _ = self._min_value(game.forecast_move(move), depth - 1)
            if v > value:
                value, max_move = v, move

        return value, max_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            while time_left() > self.TIMER_THRESHOLD:
                best_move = self.alphabeta(game, self.search_depth, -inf, inf)
                self.search_depth += 1
            if best_move == (-1, -1):
                legal_moves = game.get_legal_moves(self)

                if len(legal_moves) > 0:
                    best_move = game.get_legal_moves(self)[0]
            return best_move

        except SearchTimeout:
            if best_move == (-1, -1):
                legal_moves = game.get_legal_moves(self)

                if len(legal_moves) > 0:
                    best_move = game.get_legal_moves(self)[0]
            return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return self._max_value(game, depth, -inf, inf)[1]

    def _check_timeout(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def _reaches_terminal(self, legal_moves, depth):
        self._check_timeout()
        return (len(legal_moves) <= 0) or (depth <= 0)

    def _min_value(self, game, depth, alpha, beta):
        self._check_timeout()
        legal_moves = game.get_legal_moves(game.active_player)
        if self._reaches_terminal(game.get_legal_moves(game.active_player), depth):
            return self.score(game, self), None

        value = inf
        min_move = None
        for move in legal_moves:
            v, _ = self._max_value(game.forecast_move(move), depth - 1, alpha, beta)
            beta = min(beta, v)
            if v < value:
                value, min_move = v, move
            if v <= alpha:
                break

        return value, min_move

    def _max_value(self, game, depth, alpha, beta):
        self._check_timeout()
        legal_moves = game.get_legal_moves(game.active_player)
        if self._reaches_terminal(game.get_legal_moves(game.active_player), depth):
            return self.score(game, self), None

        value = -inf
        max_move = None
        for move in legal_moves:
            v, _ = self._min_value(game.forecast_move(move), depth - 1, alpha, beta)
            alpha = max(alpha, v)
            if v > value:
                value, max_move = v, move
            if v >= beta:
                break

        return value, max_move
