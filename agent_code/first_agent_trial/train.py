from collections import namedtuple, deque
import numpy as np
import pickle
from typing import List

import events as e
from .callbacks import state_to_features, ACTIONS, STEP

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


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
    self.q_sa = np.zeros((STEP, len(ACTIONS)))
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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    reward = reward_from_events(self, events)
    fit_models(self, old_game_state, self_action, new_game_state, reward)
    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                   reward_from_events(self, events)))


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
    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_LEFT: 0.0005,
        e.MOVED_RIGHT: 0.0005,
        e.MOVED_UP: 0.0005,
        e.MOVED_DOWN: 0.0005,
        e.WAITED: -0.001,
        e.INVALID_ACTION: -0.03,
        e.BOMB_EXPLODED: 0.002,
        e.BOMB_DROPPED: 0.0003,
        e.CRATE_DESTROYED: 0.05,
        e.COIN_FOUND: 0.06,
        e.COIN_COLLECTED: 0.3,
        e.KILLED_OPPONENT: 0.5,
        e.KILLED_SELF: -0.2,
        e.GOT_KILLED: -0.1,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 0.1,
        # PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def fit_models(self, old_game_state, action, new_game_state, reward):
    old_state_idx = state_to_features(old_game_state)

    new_state_idx = state_to_features(new_game_state)

    model_a_new_q_values = self.q_sa[new_state_idx, :]

    model_a_old_q_value = self.q_sa[old_state_idx, ACTIONS.index(action)]

    old_q_value = self.alpha * (
            reward + self.gamma * model_a_new_q_values[np.argmax(model_a_new_q_values)] -
            model_a_old_q_value)

    for i in range(STEP):
        for j in range(len(ACTIONS)):
            self.q_sa[i, j] += old_q_value
