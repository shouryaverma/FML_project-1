import logging
import sys
import os

import numpy as np
import pickle

from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

POSSIBLE_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = 'weights.pt'
modelpath = os.path.join(os.getcwd(), MODEL_FILE)

EXPLORATION_RATE = 0.2
MIN_EXPLORATION_RATE = 0.1
EXPLORATION_RATE_DECAY = 0.99


def setup(self):
    self.is_fit = False
    self.last_act_exploration = False

    if self.train:
        self.exploration_rate = EXPLORATION_RATE
    else:
        self.exploration_rate = MIN_EXPLORATION_RATE

    if not self.train:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    # file_handler = logging.FileHandler(f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    # file_handler.setLevel(logging.DEBUG)
    # self.logger.addHandler(file_handler)
    if self.train and not os.path.isfile(os.path.join(os.getcwd(), MODEL_FILE)):
        self.logger.info("Setting up model from scratch.")
        print("Setting up model from scratch.")
        self.model_a = MultiOutputRegressor(LGBMRegressor(zero_as_missing=True, use_missing=False))

    elif self.train and os.path.isfile(os.path.join(os.getcwd(), MODEL_FILE)):
        print("Loading model from saved state.")
        with open(modelpath, "rb") as file:
            self.model_a = pickle.load(file)

    elif not self.train and os.path.exists(MODEL_FILE):
        self.logger.info("Using existing model to play")
        with open(modelpath, "rb") as file:
            self.model_a = pickle.load(file)

        self.is_fit = True

    else:
        self.logger.info("Loading model from saved state.")
        print("Loading model from saved state.")
        with open(MODEL_FILE, "rb") as file:
            self.model_a = pickle.load(file)


def act(self, game_state: dict) -> str:
    if not self.is_fit or np.random.rand() < self.exploration_rate:
        # explore
        self.logger.debug("Exploring random action")
        action = np.random.choice(POSSIBLE_ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        self.last_act_exploration = True
    else:
        # exploit
        model = self.model if not self.train else self.model_a
        self.logger.debug("Exploiting (predict actions)")
        action = POSSIBLE_ACTIONS[np.argmax(model.predict(state_to_features(game_state)))]
        self.last_act_exploration = False

    self.exploration_rate *= EXPLORATION_RATE_DECAY
    self.exploration_rate = max(self.exploration_rate, 0.1)

    self.logger.debug(f"Took action {action}")
    return action


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

    self_cor_channel = np.array(game_state["self"][3])
    down_channel = [self_cor_channel[1] + 1, self_cor_channel[0]]
    up_channel = [self_cor_channel[1] - 1, self_cor_channel[0]]
    left_channel = [self_cor_channel[1], self_cor_channel[0] - 1]
    right_channel = [self_cor_channel[1], self_cor_channel[0] + 1]

    left = cor_states(game_state, left_channel, 'left')
    right = cor_states(game_state, right_channel, 'right')
    up = cor_states(game_state, up_channel, 'up')
    down = cor_states(game_state, down_channel, 'down')
    # self_ch = cor_states(game_state, self_cor_channel) # no need to check current position

    stacked_channels = np.array([left] + [right] + [up] + [down] + [game_state["self"][2]])
    stacked_channels = np.concatenate(stacked_channels, axis=None)
    # and return them as a vector
    return stacked_channels.reshape(1, -1)


def cor_states(game_state, coordinates, direction_str):
    """
        Checks the given coordinates if there is a coin, if there is a wall or crate or free, and if there is a danger.
        Danger also contains walls as well.

        :param game_state:  A dictionary describing the current game board.
        :param coordinates:  The given coordinates.
        :return: np.array of size 3 -> ([0 or 1], [0 or 1], [0 or 1])
        """
    field_channel = np.array(game_state["field"]).T
    coins = np.array(game_state["coins"]).T
    bombs = np.array(game_state["bombs"]).T
    explosion = np.array(game_state["explosion_map"])
    state_bits = np.zeros(4)
    if field_channel[coordinates[0], coordinates[1]] == 1:
        state_bits[1] = 1
    elif field_channel[coordinates[0], coordinates[1]] == 0:
        state_bits[1] = 0
    else:
        state_bits[2] = 1
    if coins.size != 0:
        for ind in range(len(coins[0])):
            if coins[0, ind] == coordinates[1] and coins[1, ind] == coordinates[0]:
                state_bits[0] = 1
    if bombs.size != 0:
        for idx in range(len(bombs[0])):
            check = list(bombs[0][idx])
            if bombs[1][idx] < 1:
                if (check[0] == coordinates[1] and np.abs(check[1] - coordinates[0]) < 3) or (
                        check[1] == coordinates[0] and np.abs(check[0] - coordinates[1]) < 3):
                    state_bits[2] = 1
            elif bombs[1][idx] == 1:
                if (check[0] == coordinates[1] and np.abs(check[1] - coordinates[0]) < 2) or (
                        check[1] == coordinates[0] and np.abs(check[0] - coordinates[1]) < 2):
                    state_bits[2] = 1
            if check[0] == coordinates[1] and check[1] == coordinates[0]:
                state_bits[2] = 1
    if explosion[coordinates[1], coordinates[0]] != 0:
        state_bits[2] = 1

    if direction_str == 'left' and (coordinates[1] > 2) and (
            (field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[
                coordinates[0], coordinates[1] - 1] != 0 and field_channel[
                 coordinates[0] + 1, coordinates[1]] != 0) or (
                    field_channel[coordinates[0] - 1, coordinates[1]] != 0 and
                    field_channel[
                        coordinates[0] + 1, coordinates[1]] != 0 and
                    field_channel[coordinates[0] - 1, coordinates[1] - 1] != 0 and
                    field_channel[coordinates[0], coordinates[1] - 2] != 0 and field_channel[
                        coordinates[0] + 1, coordinates[1] - 1] != 0)
            or (
                    field_channel[coordinates[0] - 1, coordinates[1]] != 0 and
                    field_channel[
                        coordinates[0] + 1, coordinates[1]] != 0 and
                    field_channel[coordinates[0] - 1, coordinates[1] - 1] != 0 and field_channel[
                        coordinates[0] + 1, coordinates[1] - 1] != 0 and
                    field_channel[coordinates[0] - 1, coordinates[1] - 2] != 0 and
                    field_channel[coordinates[0], coordinates[1] - 3] != 0 and field_channel[
                        coordinates[0] + 1, coordinates[1] - 2] != 0)):
        state_bits[3] = 1
    elif direction_str == 'left' and (coordinates[1] == 1 or coordinates[1] == 2) and (
            field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[
        coordinates[0], coordinates[1] - 1] != 0 and field_channel[
                coordinates[0] + 1, coordinates[1]] != 0):
        state_bits[3] = 1
    elif direction_str == 'right' and (coordinates[1] < 14) and (
            (field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[
                coordinates[0], coordinates[1] + 1] != 0 and field_channel[
                 coordinates[0] + 1, coordinates[1]] != 0) or (
                    field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[
                coordinates[0] + 1, coordinates[1]] != 0 and
                    field_channel[coordinates[0] - 1, coordinates[1] + 1] != 0 and
                    field_channel[coordinates[0], coordinates[1] + 2] != 0 and field_channel[
                        coordinates[0] + 1, coordinates[1] + 1] != 0)
            or (
                    field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[
                coordinates[0] + 1, coordinates[1]] != 0 and
                    field_channel[coordinates[0] - 1, coordinates[1] + 1] != 0 and field_channel[
                        coordinates[0] + 1, coordinates[1] + 1] != 0 and
                    field_channel[coordinates[0] - 1, coordinates[1] + 2] != 0 and
                    field_channel[coordinates[0], coordinates[1] + 3] != 0 and field_channel[
                        coordinates[0] + 1, coordinates[1] + 2] != 0)):
        state_bits[3] = 1
    elif direction_str == 'right' and (coordinates[1] == 15 or coordinates[1] == 14) and (
            field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[
        coordinates[0], coordinates[1] + 1] != 0 and field_channel[
                coordinates[0] + 1, coordinates[1]] != 0):
        state_bits[3] = 1
    elif direction_str == 'up' and (coordinates[0] > 2) and (
            (field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[
                coordinates[0], coordinates[1] + 1] != 0 and field_channel[
                 coordinates[0], coordinates[1] - 1] != 0) or (field_channel[
                                                                   coordinates[0], coordinates[1] + 1] != 0 and
                                                               field_channel[
                                                                   coordinates[0], coordinates[1] - 1] != 0 and
                                                               field_channel[
                                                                   coordinates[0] - 1, coordinates[1] + 1] != 0 and
                                                               field_channel[
                                                                   coordinates[0] - 2, coordinates[1]] != 0 and
                                                               field_channel[
                                                                   coordinates[0] - 1, coordinates[1] - 1] != 0)
            or (
                    field_channel[
                        coordinates[0], coordinates[1] + 1] != 0 and
                    field_channel[
                        coordinates[0], coordinates[1] - 1] != 0 and
                    field_channel[
                        coordinates[0] - 1, coordinates[1] + 1] != 0 and
                    field_channel[
                        coordinates[0] - 1, coordinates[1] - 1] != 0 and
                    field_channel[coordinates[0] - 2, coordinates[1] + 1] != 0 and
                    field_channel[coordinates[0] - 3, coordinates[1]] != 0 and field_channel[
                        coordinates[0] - 2, coordinates[1] - 1] != 0)):
        state_bits[3] = 1
    elif direction_str == 'up' and (coordinates[0] == 1 or coordinates[0] == 2) and (
            field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[
        coordinates[0], coordinates[1] + 1] != 0 and field_channel[coordinates[0], coordinates[1] - 1] != 0):
        state_bits[3] = 1
    elif direction_str == 'down' and (coordinates[0] < 14) and (
            (field_channel[coordinates[0] + 1, coordinates[1]] != 0 and field_channel[
                coordinates[0], coordinates[1] + 1] != 0 and field_channel[
                 coordinates[0], coordinates[1] - 1] != 0) or (field_channel[
                                                                   coordinates[0], coordinates[1] + 1] != 0 and
                                                               field_channel[
                                                                   coordinates[0], coordinates[1] - 1] != 0 and
                                                               field_channel[
                                                                   coordinates[0] + 1, coordinates[1] + 1] != 0 and
                                                               field_channel[
                                                                   coordinates[0] + 2, coordinates[1]] != 0 and
                                                               field_channel[
                                                                   coordinates[0] + 1, coordinates[1] - 1] != 0)
            or (field_channel[
                    coordinates[0], coordinates[1] + 1] != 0 and field_channel[
                    coordinates[0], coordinates[1] - 1] != 0 and
                field_channel[coordinates[0] + 1, coordinates[1] + 1] != 0 and field_channel[
                    coordinates[0] + 1, coordinates[1] - 1] != 0 and
                field_channel[coordinates[0] + 2, coordinates[1] + 1] != 0 and
                field_channel[coordinates[0] + 3, coordinates[1]] != 0 and field_channel[
                    coordinates[0] + 2, coordinates[1] - 1] != 0)):
        state_bits[3] = 1
    elif direction_str == 'down' and (coordinates[0] == 15 or coordinates[0] == 14) and (
            field_channel[coordinates[0] + 1, coordinates[1]] != 0 and field_channel[
        coordinates[0], coordinates[1] + 1] != 0 and field_channel[coordinates[0], coordinates[1] - 1] != 0):
        state_bits[3] = 1

    return state_bits
