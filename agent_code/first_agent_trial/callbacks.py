import os
import pickle
import random

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
STATE = 13
EXPLORATION_RATE = 0.02


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.exploration_rate = EXPLORATION_RATE
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    q = self.q_sa
    cond = np.max(list(q[features_to_index(state_to_features(game_state))].values()))
    if cond < self.exploration_rate:
        # explore
        cordin = np.array(game_state['self'][3])
        x = cordin[0]
        y = cordin[1]
        arena = np.array(game_state['field'])
        bomb_map = np.ones(arena.shape) * 5
        bomb_xys = [xy for (xy, t) in game_state['bombs']]

        for (xb, yb), t in game_state['bombs']:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                    bomb_map[j, i] = min(bomb_map[i, j], t)

        self.logger.debug("Exploring random action")
        # action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

        # determine valid actions
        directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid_tiles, valid_actions = [], []
        for d in directions:
            if ((arena[d] == 0) and
                    (game_state['explosion_map'][d] <= 1) and
                    (bomb_map[d] > 0) and
                    (not d in bomb_xys)):
                valid_tiles.append(d)
        if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
        if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
        if (x, y - 1) in valid_tiles: valid_actions.append('UP')
        if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
        if (x, y) in valid_tiles: valid_actions.append('WAIT')
        if game_state['self'][2]: valid_actions.append('BOMB')

        self.logger.debug(f'Valid actions: {valid_actions}')
        if len(valid_actions) == 0:
            action = 'WAIT'
        else:
            action = np.random.choice(valid_actions)
    else:
        # exploit
        self.logger.debug("Exploiting (predict actions)")
        self.logger.debug(f"Q: {list(q[features_to_index(state_to_features(game_state))].values())}")
        self.logger.debug(f"Index: {features_to_index(state_to_features(game_state))}")
        action = ACTIONS[np.argmax(list(q[features_to_index(state_to_features(game_state))].values()))]

    self.logger.debug(f"Took action {action}")
    return action

    # todo Exploration vs exploitation
    """"
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=self.model)"""


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    # channels = []
    # channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    # stacked_channels = np.stack(channels)

    self_cor_channel = np.array(game_state["self"][3])
    down_channel = [self_cor_channel[0] + 1, self_cor_channel[1]]
    up_channel = [self_cor_channel[0] - 1, self_cor_channel[1]]
    left_channel = [self_cor_channel[0], self_cor_channel[1] - 1]
    right_channel = [self_cor_channel[0], self_cor_channel[1] + 1]

    left = cor_states(game_state, left_channel)
    right = cor_states(game_state, right_channel)
    up = cor_states(game_state, up_channel)
    down = cor_states(game_state, down_channel)
    # self_ch = cor_states(game_state, self_cor_channel) # no need to check current position

    stacked_channels = np.array([left] + [right] + [up] + [down] + [game_state["self"][2]])
    stacked_channels = np.concatenate(stacked_channels, axis=None)
    # and return them as a vector
    return stacked_channels


def cor_states(game_state, coordinates):
    """
        Checks the given coordinates if there is a coin, if there is a wall or crate or free, and if there is a danger.
        Danger also contains walls as well.

        :param game_state:  A dictionary describing the current game board.
        :param coordinates:  The given coordinates.
        :return: np.array of size 3 -> ([0 or 1], [0 or 1], [0 or 1])
        """
    field_channel = np.array(game_state["field"])
    coins = np.array(game_state["coins"]).T
    bombs = np.array(game_state["bombs"]).T
    explosion = np.array(game_state["explosion_map"])
    state_bits = np.zeros(3)
    if field_channel[coordinates[0], coordinates[1]] == 1:
        state_bits[1] = 1
    elif field_channel[coordinates[0], coordinates[1]] == 0:
        state_bits[1] = 0
    else:
        state_bits[2] = 1
    if coins.size != 0:
        for ind in range(len(coins[0])):
            if coins[0, ind] == coordinates[0] and coins[1, ind] == coordinates[1]:
                state_bits[0] = 1
    if bombs.size != 0:
        for idx in range(len(bombs[0])):
            check = list(bombs[idx][0])
            if bombs[1] == 1:
                if check[0] == coordinates[0] or check[1] == coordinates[1]:
                    state_bits[2] = 1
    if explosion[coordinates[0], coordinates[1]] != 0:
        state_bits[2] = 1
    return state_bits


def features_to_index(features):
    return 2 ** 12 * features[0] + 2 ** 11 * features[1] + 2 ** 10 * features[2] + 2 ** 9 * features[3] + 2 ** 8 * \
           features[4] + \
           2 ** 7 * features[5] + 2 ** 6 * features[6] + 2 ** 5 * features[7] + 2 ** 4 * features[8] + 2 ** 3 * \
           features[9] + 2 ** 2 * features[10] + \
           2 ** 1 * features[11] + 2 ** 0 * features[12]


def get_explosion_xys(start, map, bomb_power=3):
    """
    returns all tiles hit by an explosion starting at start given a 2d map of the game
       where walls are indicated by -1
    """
    x, y = start
    expl = [(x, y)]
    for i in range(1, bomb_power + 1):
        if map[x + i, y] == -1: break
        expl.append((x + i, y))
    for i in range(1, bomb_power + 1):
        if map[x - i, y] == -1: break
        expl.append((x - i, y))
    for i in range(1, bomb_power + 1):
        if map[x, y + i] == -1: break
        expl.append((x, y + i))
    for i in range(1, bomb_power + 1):
        if map[x, y - i] == -1: break
        expl.append((x, y - i))

    return np.array(expl)
