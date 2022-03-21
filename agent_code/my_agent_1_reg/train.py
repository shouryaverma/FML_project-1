import os
import pickle
import shutil
from collections import namedtuple, deque
from datetime import datetime
from typing import List
from . import callbacks
import numpy as np

import events as e
from .callbacks import state_to_features, MODEL_FILE, POSSIBLE_ACTIONS

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from lightgbm import LGBMRegressor

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'last_act_was_exploration'))

DECREASED_DISTANCE = "DIST_DECREASED"
INCREASED_DISTANCE = "DIST_INCREASED"
BOMB_DROPPED_CORNER = "BOMB_CORNER"
BOMB_NEAR_CRATE = 'BOMB_NEAR_CRATE'
BOMB_NOT_NEAR_CRATE = 'BOMB_NOT_NEAR_CRATE'
MOVED_AWAY_FROM_BOMB = 'MOVED_AWAY_FROM_BOMB'
MOVED_TO_BOMB = 'MOVED_TO_BOMB'
DEAD_END = 'DEAD_END'
NOT_DEAD_END = 'NOT_DEAD_END'

TRAINING_SET_LEN = 10000

LEARNING_RATE = 0.5
LEARNING_RATE_DECAY = 0.95
DISCOUNT = 0.75
LAMBDA = 0.5

modelname = "weights.pt"
modelpath = os.path.join(os.getcwd(), modelname)


def setup_training(self):
    self.logger.debug("Training setup")
    self.logger.info("Training mode")

    self.learning_rate = LEARNING_RATE
    self.transitions = deque(maxlen=TRAINING_SET_LEN)

    """if os.path.exists(MODEL_FILE):
        shutil.copy(MODEL_FILE, f"./weights_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt")"""


def add_events_from_state(old_game_state, new_game_state, self_action, events):
    if old_game_state is None and self_action == 'BOMB':
        events.append(BOMB_DROPPED_CORNER)
    if old_game_state is not None:
        events = bomb_check(new_game_state, self_action, events)
        if np.array(new_game_state['coins']).T.size > 0:
            events = coin_dist_check(old_game_state, new_game_state, events)
        if np.array(new_game_state['bombs']).T.size > 0:
            events = bomb_dist_check(old_game_state, new_game_state, events)
            events = dead_end_check2(old_game_state, new_game_state, events)

    return events


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(
        f'DURING ROUND: Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if self_action is None or old_game_state is None or new_game_state is None:
        return

    events = add_events_from_state(old_game_state, new_game_state, self_action, events)
    reward = reward_from_events(self, events)

    self.transitions.append(Transition(old_game_state, self_action, new_game_state, reward, self.last_act_exploration))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'END OF ROUND: Encountered event(s) {", ".join(map(repr, events))}')

    reward = reward_from_events(self, events)
    self.transitions.append(Transition(last_game_state, last_action, None, reward, self.last_act_exploration))

    fit_models(self)

    self.learning_rate *= LEARNING_RATE_DECAY
    self.learning_rate = max(self.learning_rate, 0.1)

    """with open(MODEL_FILE, "wb") as file:
        pickle.dump(self.model_a, file)"""
    with open(MODEL_FILE, 'wb') as f:
        np.save(f, self.model_a)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.WAITED: -0.1,
        e.INVALID_ACTION: -0.8,
        e.BOMB_EXPLODED: 0.14,
        e.BOMB_DROPPED: -0.1,
        # e.CRATE_DESTROYED: 0.02,
        # e.COIN_FOUND: 0.02,
        e.COIN_COLLECTED: 0.6,
        e.KILLED_OPPONENT: 1,
        e.KILLED_SELF: -1,
        e.GOT_KILLED: -1,
        e.OPPONENT_ELIMINATED: 2,
        e.SURVIVED_ROUND: 0.2,
        # custom events
        e.DECREASED_DISTANCE: 0.2,
        e.INCREASED_DISTANCE: -0.2,
        e.BOMB_DROPPED_CORNER: -0.8,
        e.BOMB_NEAR_CRATE: 0.6,
        e.BOMB_NOT_NEAR_CRATE: -0.4,
        e.MOVED_AWAY_FROM_BOMB: 0.5,
        e.MOVED_TO_BOMB: -0.6,
        e.DEAD_END: -0.6,
        e.NOT_DEAD_END: 0.6
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def fit_models(self):
    features = []
    targets_a = []
    eligibility_traces = np.zeros(len(POSSIBLE_ACTIONS))

    for idx, [previous_game_state, action, game_state, reward, last_act_was_exploration] in enumerate(self.transitions):
        old_state_features = state_to_features(previous_game_state)
        features.append(old_state_features.reshape(-1))

        if self.is_fit:
            if game_state is not None:
                new_state_features = state_to_features(game_state)

                model_a_new_q_values = self.model_a.predict(new_state_features)

                model_a_old_q_value = self.model_a.predict(old_state_features)[0][POSSIBLE_ACTIONS.index(action)]
                self.logger.debug(f'Q table: {model_a_new_q_values[0][np.argmax(model_a_new_q_values)]}')

                q_update_a = self.learning_rate * (
                        reward + DISCOUNT * model_a_new_q_values[0][
                    np.argmax(model_a_new_q_values)] - model_a_old_q_value)
            else:
                model_a_old_q_value = self.model_a.predict(old_state_features)

                q_update_a = self.learning_rate * (
                        reward - model_a_old_q_value[0][POSSIBLE_ACTIONS.index(action)])
        else:
            model_a_new_q_values = np.zeros(len(POSSIBLE_ACTIONS)).reshape(1, -1)

            q_update_a = self.learning_rate * reward

        eligibility_traces[POSSIBLE_ACTIONS.index(action)] += 1

        for i in range(len(POSSIBLE_ACTIONS)):
            model_a_new_q_values[0][i] += (q_update_a * eligibility_traces[i])

            if last_act_was_exploration:
                eligibility_traces[i] = 0
            else:
                eligibility_traces[i] *= (DISCOUNT * LAMBDA)

        targets_a.append(model_a_new_q_values[0])

    self.model_a.fit(features, normalize(targets_a))
    self.is_fit = True


def coin_dist_check(old_game_state, new_game_state, events):
    old_coins = np.array(old_game_state["coins"]).T
    new_coins = np.array(new_game_state["coins"]).T
    old_self_cor_channel = np.array(old_game_state["self"][3]).T
    new_self_cor_channel = np.array(new_game_state["self"][3]).T
    old_cor_turned = old_self_cor_channel.T
    if old_coins.size != 0 and new_coins.size != 0:
        if old_coins.size > new_coins.size:
            max_coin = old_coins
            min_coin = new_coins
        else:
            max_coin = new_coins
            min_coin = old_coins
        for ind in range(len(max_coin[0])):
            old_dist_coin = np.sum(np.abs(max_coin[:, ind] - old_cor_turned))
            if old_dist_coin < 5:
                for index in range(len(min_coin[0])):
                    if max_coin[0, ind] == min_coin[0, index] and max_coin[1, ind] == min_coin[1, index]:
                        max_dist = np.abs((max_coin[0, ind] - old_self_cor_channel[1])) + np.abs(
                            (max_coin[1, ind] - old_self_cor_channel[0]))
                        min_dist = np.abs((min_coin[0, index] - new_self_cor_channel[1])) + np.abs(
                            (min_coin[1, index] - new_self_cor_channel[0]))
                        if min_dist < max_dist and max_coin.all() == old_coins.all():
                            events.append(DECREASED_DISTANCE)
                        elif min_dist > max_dist and max_coin.all() == old_coins.all():
                            events.append(INCREASED_DISTANCE)
                        elif min_dist < max_dist and min_coin.all() == old_coins.all():
                            events.append(INCREASED_DISTANCE)
                        elif min_dist > max_dist and min_coin.all() == old_coins.all():
                            events.append(DECREASED_DISTANCE)
    return events


def bomb_check(new_game_state, self_action, events):
    self_cord = np.array(new_game_state["self"][3])
    if (self_cord[1] == 1 and self_cord[0] == 1) or (self_cord[1] == 1 and self_cord[0] == 15) or \
            (self_cord[1] == 15 and self_cord[0] == 1) or (self_cord[1] == 15 and self_cord[0] == 15):
        if self_action == 'BOMB':
            events.append(BOMB_DROPPED_CORNER)
    field = np.array(new_game_state["field"]).T
    if (field[self_cord[1], self_cord[0] + 1] == 1) or (field[self_cord[1], self_cord[0] - 1] == 1) or \
            (field[self_cord[1] + 1, self_cord[0]] == 1) or (field[self_cord[1] - 1, self_cord[0]] == 1):
        if self_action == 'BOMB':
            events.append(BOMB_NEAR_CRATE)
    else:
        if self_action == 'BOMB':
            events.append(BOMB_NOT_NEAR_CRATE)
    return events


def bomb_dist_check(old_game_state, new_game_state, events):
    old_self_cor_channel = np.array(old_game_state["self"][3])
    new_self_cor_channel = np.array(new_game_state["self"][3])
    for bomb in old_game_state['bombs']:
        bomb_location = bomb[0]

        old_dist_bomb = np.sum(np.abs(bomb_location - old_self_cor_channel))
        new_dist_bomb = np.sum(np.abs(bomb_location - new_self_cor_channel))
        if old_dist_bomb < 5:
            if new_dist_bomb > old_dist_bomb:
                events.append('MOVED_AWAY_FROM_BOMB')
            elif new_dist_bomb < old_dist_bomb:
                events.append('MOVED_TO_BOMB')
    return events


def dead_end_check2(old_game_state, new_game_state, events):
    new_own_location = np.asarray(new_game_state['self'][-1])
    old_own_location = np.asarray(old_game_state['self'][-1])
    feat = state_to_features(old_game_state)
    chosen_direction = new_own_location - old_own_location
    turn_direction = chosen_direction[::-1]
    for bomb in old_game_state['bombs']:
        bomb_location = bomb[0]
        new_dist_bomb = np.sum(np.abs(bomb_location - new_own_location))
        if new_dist_bomb < 5:
            if np.sum(chosen_direction) != 0:
                if (turn_direction == [1, 0]).all() and feat[0][15] == 1 and (bomb_location == new_own_location).any():
                    events.append('DEAD_END')
                elif (turn_direction == [-1, 0]).all() and feat[0][11] == 1 and (bomb_location == new_own_location).any():
                    events.append('DEAD_END')
                elif (turn_direction == [0, 1]).all() and feat[0][7] == 1 and (bomb_location == new_own_location).any():
                    events.append('DEAD_END')
                elif (turn_direction == [0, -1]).all() and feat[0][3] == 1 and (bomb_location == new_own_location).any():
                    events.append('DEAD_END')
                else:
                    events.append('NOT_DEAD_END')
            elif (turn_direction == [0, 0]).all() and (bomb_location != new_own_location).all():
                events.append('NOT_DEAD_END')
    return events
