import os
import pickle
import numpy as np
from collections import deque
from random import shuffle

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
STATE = 17
EXPLORATION_RATE = 0.1
EXPLORE_COUNT = 0
EXPLOIT_COUNT = 0


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

    self.logger.info("Loading Q_sa_inc_3 function from saved state.")
    with open('Q_sa_rule.npy', 'rb') as f:
        self.q_sa = np.load(f, allow_pickle=True)
    self.q_sa = self.q_sa.tolist()

    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    global EXPLORE_COUNT
    global EXPLOIT_COUNT
    q = self.q_sa
    cond = np.max(list(q[features_to_index(state_to_features(game_state))].values()))
    if cond < self.exploration_rate:
        # explore
        cordin = np.array(game_state['self'][3])
        x = cordin[1]
        y = cordin[0]
        arena = np.array(game_state['field']).T
        bomb_map = np.ones(arena.shape) * 5
        bomb_xys = [xy for (xy, t) in game_state['bombs']]
        if bomb_xys:
            bomb_xys = [(bomb_xys[0][1], bomb_xys[0][0])]

        for (yb, xb), t in game_state['bombs']:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                    bomb_map[i, j] = min(bomb_map[i, j], t)

        self.logger.debug("Exploring random action")
        self.logger.debug(f"Random Q: {list(q[features_to_index(state_to_features(game_state))].values())}")
        self.logger.debug(f"Explore index: {features_to_index(state_to_features(game_state))}")

        # determine valid actions
        directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid_tiles, valid_actions = [], []
        for d in directions:
            d_trans = (d[1], d[0])
            if ((arena[d] == 0) and
                    (game_state['explosion_map'][d_trans] < 1) and
                    (bomb_map[d] > 0) and
                    (not d in bomb_xys)):
                valid_tiles.append(d)
        if (x - 1, y) in valid_tiles: valid_actions.append('UP')
        if (x + 1, y) in valid_tiles: valid_actions.append('DOWN')
        if (x, y - 1) in valid_tiles: valid_actions.append('LEFT')
        if (x, y + 1) in valid_tiles: valid_actions.append('RIGHT')
        if (x, y) in valid_tiles: valid_actions.append('WAIT')
        if game_state['self'][2]: valid_actions.append('BOMB')

        self.logger.debug(f'Valid actions: {valid_actions}')
        if len(valid_actions) == 0:
            action = 'WAIT'
            EXPLORE_COUNT += 1
        else:
            max_act = ACTIONS[np.argmax(list(q[features_to_index(state_to_features(game_state))].values()))]
            if max_act in valid_actions and cond > 0:
                EXPLOIT_COUNT += 1
                action = max_act
            else:
                EXPLORE_COUNT += 1
                action = np.random.choice(valid_actions)
        """arena = game_state['field']
        _, score, bombs_left, (x, y) = game_state['self']
        bombs = game_state['bombs']
        bomb_xys = [xy for (xy, t) in bombs]
        others = [xy for (n, s, b, xy) in game_state['others']]
        coins = game_state['coins']
        bomb_map = np.ones(arena.shape) * 5
        for (xb, yb), t in bombs:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                    bomb_map[i, j] = min(bomb_map[i, j], t)

        # If agent has been in the same location three times recently, it's a loop
        if self.coordinate_history.count((x, y)) > 2:
            self.ignore_others_timer = 5
        else:
            self.ignore_others_timer -= 1
        self.coordinate_history.append((x, y))

        self.logger.debug("Exploring random action")
        self.logger.debug(f"Random Q: {list(q[features_to_index(state_to_features(game_state))].values())}")
        self.logger.debug(f"Explore index: {features_to_index(state_to_features(game_state))}")

        # Check which moves make sense at all
        directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid_tiles, valid_actions = [], []
        for d in directions:
            if ((arena[d] == 0) and
                    (game_state['explosion_map'][d] < 1) and
                    (bomb_map[d] > 0) and
                    (not d in others) and
                    (not d in bomb_xys)):
                valid_tiles.append(d)
        if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
        if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
        if (x, y - 1) in valid_tiles: valid_actions.append('UP')
        if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
        if (x, y) in valid_tiles: valid_actions.append('WAIT')
        # Disallow the BOMB action if agent dropped a bomb in the same spot recently
        if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
        self.logger.debug(f'Valid actions: {valid_actions}')

        # Collect basic action proposals in a queue
        # Later on, the last added action that is also valid will be chosen
        action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        shuffle(action_ideas)

        # Compile a list of 'targets' the agent should head towards
        cols = range(1, arena.shape[0] - 1)
        rows = range(1, arena.shape[0] - 1)
        dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                     and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
        crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
        targets = coins + dead_ends + crates
        # Add other agents as targets if in hunting mode or no crates/coins left
        if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
            targets.extend(others)

        # Exclude targets that are currently occupied by a bomb
        targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

        # Take a step towards the most immediately interesting target
        free_space = arena == 0
        if self.ignore_others_timer > 0:
            for o in others:
                free_space[o] = False
        d = look_for_targets(free_space, (x, y), targets, self.logger)
        if d == (x, y - 1): action_ideas.append('UP')
        if d == (x, y + 1): action_ideas.append('DOWN')
        if d == (x - 1, y): action_ideas.append('LEFT')
        if d == (x + 1, y): action_ideas.append('RIGHT')
        if d is None:
            self.logger.debug('All targets gone, nothing to do anymore')
            action_ideas.append('WAIT')

        # Add proposal to drop a bomb if at dead end
        if (x, y) in dead_ends:
            action_ideas.append('BOMB')
        # Add proposal to drop a bomb if touching an opponent
        if len(others) > 0:
            if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
                action_ideas.append('BOMB')
        # Add proposal to drop a bomb if arrived at target and touching crate
        if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
            action_ideas.append('BOMB')

        # Add proposal to run away from any nearby bomb about to blow
        for (xb, yb), t in bombs:
            if (xb == x) and (abs(yb - y) < 4):
                # Run away
                if (yb > y): action_ideas.append('UP')
                if (yb < y): action_ideas.append('DOWN')
                # If possible, turn a corner
                action_ideas.append('LEFT')
                action_ideas.append('RIGHT')
            if (yb == y) and (abs(xb - x) < 4):
                # Run away
                if (xb > x): action_ideas.append('LEFT')
                if (xb < x): action_ideas.append('RIGHT')
                # If possible, turn a corner
                action_ideas.append('UP')
                action_ideas.append('DOWN')
        # Try random direction if directly on top of a bomb
        for (xb, yb), t in bombs:
            if xb == x and yb == y:
                action_ideas.extend(action_ideas[:4])
        action = 'WAIT'
        EXPLORE_COUNT += 1
        # Pick last action added to the proposals list that is also valid
        while len(action_ideas) > 0:
            a = action_ideas.pop()
            if a in valid_actions:
                # Keep track of chosen action for cycle detection
                if a == 'BOMB':
                    self.bomb_history.append((x, y))

                action = a
                break"""
    else:
        # exploit
        EXPLOIT_COUNT += 1
        self.logger.debug("Exploiting (predict actions)")
        self.logger.debug(f"Q: {list(q[features_to_index(state_to_features(game_state))].values())}")
        self.logger.debug(f"Index: {features_to_index(state_to_features(game_state))}")
        action = ACTIONS[np.argmax(list(q[features_to_index(state_to_features(game_state))].values()))]

    self.logger.debug(f"Explore count: {EXPLORE_COUNT}, Exploit count: {EXPLOIT_COUNT}")
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

    # For example, you could construct several channels of equal shape, ...
    # channels = []
    # channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    # stacked_channels = np.stack(channels)

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
    return stacked_channels


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
            check = list(bombs[idx][0])
            if bombs[1] <= 1:
                if (check[0] == coordinates[1] and np.abs(check[1] - coordinates[0]) < 3) or (
                        check[1] == coordinates[0] and np.abs(
                    check[0] - coordinates[1]) < 3):
                    state_bits[2] = 1
            elif bombs[1] == 2:
                if (check[0] == coordinates[1] and np.abs(check[1] - coordinates[0]) < 2) or (
                        check[1] == coordinates[0] and np.abs(
                    check[0] - coordinates[1]) < 2):
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
                        coordinates[0] + 1, coordinates[1] - 1] != 0)):
        """or (
                    field_channel[coordinates[0] - 1, coordinates[1]] != 0 and
                    field_channel[
                        coordinates[0] + 1, coordinates[1]] != 0 and
                    field_channel[coordinates[0] - 1, coordinates[1] - 1] != 0 and field_channel[
                        coordinates[0] + 1, coordinates[1] - 1] != 0 and
                    field_channel[coordinates[0] - 1, coordinates[1] - 2] != 0 and
                    field_channel[coordinates[0], coordinates[1] - 3] != 0 and field_channel[
                        coordinates[0] + 1, coordinates[1] - 2] != 0)):"""
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
                        coordinates[0] + 1, coordinates[1] + 1] != 0)):
        """or (
                    field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[
                coordinates[0] + 1, coordinates[1]] != 0 and
                    field_channel[coordinates[0] - 1, coordinates[1] + 1] != 0 and field_channel[
                        coordinates[0] + 1, coordinates[1] + 1] != 0 and
                    field_channel[coordinates[0] - 1, coordinates[1] + 2] != 0 and
                    field_channel[coordinates[0], coordinates[1] + 3] != 0 and field_channel[
                        coordinates[0] + 1, coordinates[1] + 2] != 0)):"""
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
                                                                   coordinates[0] - 1, coordinates[1] - 1] != 0)):
        """or (
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
                        coordinates[0] - 2, coordinates[1] - 1] != 0)):"""
        state_bits[3] = 1
    elif direction_str == 'up' and (coordinates[0] == 1 or coordinates[0] == 2) and (
            field_channel[coordinates[0] - 1, coordinates[1]] != 0 and field_channel[
        coordinates[0], coordinates[1] + 1] != 0 and field_channel[coordinates[0], coordinates[1] - 1] != 0):
        state_bits[3] = 1
    elif direction_str == 'down' and (coordinates[0] < 14) and (
            (field_channel[coordinates[0] + 1, coordinates[1]] != 0 and field_channel[
                coordinates[0], coordinates[1] + 1] != 0 and field_channel[
                 coordinates[0], coordinates[1] - 1] != 0) or (field_channel[
                coordinates[0], coordinates[1] + 1] != 0 and field_channel[
                 coordinates[0], coordinates[1] - 1] != 0 and
                    field_channel[coordinates[0] + 1, coordinates[1] + 1] != 0 and
                    field_channel[coordinates[0] + 2, coordinates[1]] != 0 and field_channel[
                        coordinates[0] + 1, coordinates[1] - 1] != 0)):
        """or (field_channel[
                coordinates[0], coordinates[1] + 1] != 0 and field_channel[
                 coordinates[0], coordinates[1] - 1] != 0 and
                    field_channel[coordinates[0] + 1, coordinates[1] + 1] != 0 and field_channel[
                        coordinates[0] + 1, coordinates[1] - 1] != 0 and
                    field_channel[coordinates[0] + 2, coordinates[1] + 1] != 0 and
                    field_channel[coordinates[0] + 3, coordinates[1]] != 0 and field_channel[
                        coordinates[0] + 2, coordinates[1] + 1] != 0)):"""
        state_bits[3] = 1
    elif direction_str == 'down' and (coordinates[0] == 15 or coordinates[0] == 14) and (
            field_channel[coordinates[0] + 1, coordinates[1]] != 0 and field_channel[
        coordinates[0], coordinates[1] + 1] != 0 and field_channel[coordinates[0], coordinates[1] - 1] != 0):
        state_bits[3] = 1

    return state_bits


def features_to_index(features):
    state_no = STATE
    index = 0
    for n in range(state_no):
        index += 2 ** (state_no - n - 1) * features[n]
    return index


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


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]
