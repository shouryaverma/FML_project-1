from collections import namedtuple, deque
import numpy as np
import pickle
import os
from typing import List

import events as e
from .callbacks import state_to_features, ACTIONS, STATE, features_to_index

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
DECREASED_DISTANCE = "DIST_DECREASED"
INCREASED_DISTANCE = "DIST_INCREASED"
TOTAL_STATES = 8 ** 4 * 2


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.logger.debug("Parameters set up")
    self.alpha = 0.1
    self.gamma = 0.9

    if not os.path.isfile("Q_sa_1.npy"):
        self.logger.info("Setting up Q_sa function")
        construct_q_table(TOTAL_STATES)
        with open('Q_sa_1.npy', 'rb') as f:
            self.q_sa = np.load(f, allow_pickle=True)
    else:
        self.logger.info("Loading Q_sa function from saved state.")
        with open('Q_sa_1.npy', 'rb') as f:
            self.q_sa = np.load(f, allow_pickle=True)
    self.q_sa = self.q_sa.tolist()
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    if old_game_state is not None:
        events = coin_dist_check(old_game_state, new_game_state, events)

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    reward = reward_from_events(self, events)
    fit_models(self, old_game_state, self_action, new_game_state, reward)
    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    reward = reward_from_events(self, events)
    fit_models(self, last_game_state, last_action, None, reward)
    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, None, reward))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    with open('Q_sa_1.npy', 'wb') as f:
        np.save(f, self.q_sa)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_LEFT: 0.5,
        e.MOVED_RIGHT: 0.5,
        e.MOVED_UP: 0.5,
        e.MOVED_DOWN: 0.5,
        e.WAITED: -0.5,
        e.INVALID_ACTION: -2,
        e.BOMB_EXPLODED: 0,
        e.BOMB_DROPPED: 0.3,
        e.CRATE_DESTROYED: 3,
        e.COIN_FOUND: 10,
        e.COIN_COLLECTED: 20,
        e.KILLED_OPPONENT: 20,
        e.KILLED_SELF: -20,
        e.GOT_KILLED: -20,
        e.OPPONENT_ELIMINATED: 20,
        e.SURVIVED_ROUND: 20,
        e.DECREASED_DISTANCE: 2,
        e.INCREASED_DISTANCE: -2
        # PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def fit_models(self, old_game_state, action, new_game_state, reward):
    if old_game_state is None:
        old_state_idx = 0
    else:
        old_state_idx = features_to_index(state_to_features(old_game_state))

    if new_game_state is None:
        model_a_old_q_value = self.q_sa[old_state_idx][action]

        old_q_value = self.alpha * (
                reward - model_a_old_q_value)
    else:
        new_state_idx = features_to_index(state_to_features(new_game_state))

        model_a_new_q_values = list(self.q_sa[new_state_idx].values())
        model_a_old_q_value = self.q_sa[old_state_idx][action]

        old_q_value = self.alpha * (
                reward + self.gamma * model_a_new_q_values[np.argmax(model_a_new_q_values)] -
                model_a_old_q_value)

    self.q_sa[old_state_idx][action] += old_q_value


def construct_q_table(state_count):
    Q = {}
    for s in range(state_count):
        Q[s] = {}
        for a in ACTIONS:
            if a == "WAIT" or a == "BOMB":
                Q[s][a] = 0.1
            else:
                Q[s][a] = 0.2

    with open('Q_sa_1.npy', 'wb') as f:
        np.save(f, Q)

def coin_dist_check(old_game_state, new_game_state, events):
    old_coins = np.array(old_game_state["coins"]).T
    new_coins = np.array(new_game_state["coins"]).T
    old_self_cor_channel = np.array(old_game_state["self"][3]).T
    new_self_cor_channel = np.array(new_game_state["self"][3]).T
    if old_coins.size != 0 and new_coins.size != 0:
        if old_coins.size > new_coins.size:
            max_coin = old_coins
            min_coin = new_coins
        else:
            max_coin = new_coins
            min_coin = old_coins
        for ind in range(len(max_coin[0])):
            for index in range(len(min_coin[0])):
                if max_coin[0, ind] == min_coin[0, index] and max_coin[1, ind] == min_coin[1, index]:
                    max_dist = np.abs((max_coin[0, ind] - old_self_cor_channel[0])) + np.abs(
                        (max_coin[1, ind] - old_self_cor_channel[1]))
                    min_dist = np.abs((min_coin[0, index] - new_self_cor_channel[0])) + np.abs(
                        (min_coin[1, index] - new_self_cor_channel[1]))
                    if min_dist < max_dist and max_coin.all() == old_coins.all():
                        events.append(DECREASED_DISTANCE)
                    elif min_dist > max_dist and max_coin.all() == old_coins.all():
                        events.append(INCREASED_DISTANCE)
                    elif min_dist < max_dist and min_coin.all() == old_coins.all():
                        events.append(INCREASED_DISTANCE)
                    elif min_dist > max_dist and min_coin.all() == old_coins.all():
                        events.append(DECREASED_DISTANCE)
    return events