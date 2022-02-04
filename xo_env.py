import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment, utils
from tf_agents.environments import tf_py_environment
from tf_agents.environments import tf_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from collections import namedtuple

class XOEnv_masked(py_environment.PyEnvironment):

    #https://github.com/tensorflow/agents/issues/255
    def __init__(self, ngrid=3, player='X'):
        self.ngrid = ngrid
        self._state = np.zeros((self.ngrid, self.ngrid))
        self._episode_ended = False
        self.discount = 1

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0, maximum=(self.ngrid)**2-1,
            name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1, ngrid**2),
            dtype=np.int32,
            minimum=-1, maximum=1,
            name='board')
        self._mask_spec = array_spec.ArraySpec(
            shape=(ngrid**2, ),
            dtype=np.bool_,
            name='allowed_moves'
        )

        self.player = 1 if player == 'X' else -1
        self._Obs = namedtuple('observation', 'board allowed_moves')
        self.opponent = XOplayer(ngrid=ngrid, player='O' if player=='X' else 'X')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._Obs(self._observation_spec, self._mask_spec)

    def _reset(self):
        self._state = np.zeros(self.ngrid**2)
        self._episode_ended = False
        if self.player==-1:
            self._opponent_move()
        return ts.restart(self._make_observation())

    def _check_game_state(self):

        gstate = self._state.reshape((self.ngrid, self.ngrid))

        def check_player(player):
            # TODO for larger grid do not require full row
            for i in range(self.ngrid):
                if (gstate[i, :] == player).all():
                    return True
                if (gstate[:, i] == player).all():
                    return True

            if (np.diag(gstate) == player).all():
                return True
            if (np.diag(gstate[::-1, :]) == player).all():
                return True

            return False


        if check_player(self.player):
            return 'player_won'
        if check_player(-self.player):
            return 'player_lost'

        if np.absolute(self._state).sum() == self.ngrid**2:
            return 'draw'

        return 'continue'

    def _action_mask(self):
        if not self._episode_ended:
            return np.array(self._state==0, dtype=np.bool_)
        else:
            return np.ones(self.ngrid**2, dtype=np.bool_)


    def _make_observation(self):
        return self._Obs(np.array([self._state], dtype=np.int32), self._action_mask())

    def _opponent_move(self):
        #TODO: add proper opponent
        move = self.opponent.get_move(self._state)

        # Opponent is -I (X=1, O=-1)
        self._state[move] = -self.player

        game_state = self._check_game_state()

        if game_state == 'draw':
            r = 0
        elif game_state == 'player_lost':
            r = -1
        elif game_state == 'player_won':
            raise KeyError('Cannot win after opponetn turn!')
        elif game_state == 'continue':
            r = 0
            return ts.transition(self._make_observation(), reward=r, discount=self.discount)
        else:
            raise KeyError('Inv key')

        self._episode_ended = True
        return ts.termination(self._make_observation(), reward=r)

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()


        grid_state = self._state[action]

        if grid_state == 0:
            self._state[action] = self.player
        else:
            raise ValueError('Invalid action attempted!')

        game_state = self._check_game_state()

        if game_state == 'player_won':
            r = 1
        elif game_state == 'draw':
            r = 0
        elif game_state == 'continue':
            return self._opponent_move()
        elif game_state == 'player_lost':
            raise KeyError('Player cannot lose after own turn!')
        else:
            raise KeyError('Inv key')

        self._episode_ended = True
        return ts.termination(self._make_observation(), reward=r)



class XOplayer():

    def __init__(self,ngrid=3, player='X'):

        self.ngrid = ngrid
        self.player = 1 if player == 'X' else -1

    def _get_lines(self, board_state):

        gstate = board_state.reshape((self.ngrid, self.ngrid))
        rows = np.zeros((self.ngrid, 3))
        cols = np.zeros((self.ngrid, 3))
        diags = np.zeros((2, 3))

        for i in range(self.ngrid):
            rows[i,:] = gstate[i,:]
            cols[i,:] = gstate[:,i]

        diags[0, :] = np.diag(gstate)
        diags[1, :] = np.diag(gstate[::-1, :])

        return {'rows':rows, 'cols':cols, 'diags':diags}

    def _find_move(self, type_, board_state):
        move = None

        if type_=='winmove':
            lsum = 2
        elif type_=='blockmove':
            lsum = -2
        else:
            raise KeyError('Invalid key for move type:', type_)

        lsum *= self.player

        for type_, lines in self._get_lines(board_state).items():
            for i, line in enumerate(lines):
                if line.sum()==lsum:
                    idx = np.where(line==0)[0]
                    if type_=='rows':
                        move = [i, idx]
                    elif type_=='cols':
                        move = [idx, i]
                    elif type_=='diags':
                        if i == 0:
                            move = [idx, idx]
                        elif i==1:
                            move = [2-idx, idx]
                    else:
                        raise KeyError('Inv. key.', type_)
                    break

            if move is not None: break

        return move


    def get_move(self, board_state):
        pos_moves = np.argwhere(board_state == 0).reshape(-1)
        if len(pos_moves)==1:
            move = pos_moves[0]
        else:
            move = self._find_move('winmove', board_state)
            if move is None:
                move = self._find_move('blockmove', board_state)
            if move is None:
                move = np.random.choice(pos_moves)
            else:
                move = move[0]*self.ngrid + move[1]

        return move

class XOEnv(py_environment.PyEnvironment):

    def __init__(self, ngrid=3):
        self.ngrid = ngrid
        self._state = np.zeros((self.ngrid, self.ngrid))
        self._episode_ended = False
        self.discount = 1

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0, maximum=(self.ngrid)**2-1,
            name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1, ngrid**2),
            dtype=np.int32,
            minimum=-1, maximum=1,
            name='observation')


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.zeros(self.ngrid**2)
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _check_game_state(self):

        gstate = self._state.reshape((self.ngrid, self.ngrid))

        def check_player(player):
            # TODO for larger grid do not require full row
            for i in range(self.ngrid):
                if (gstate[i, :] == player).all():
                    return True
                if (gstate[:, i] == player).all():
                    return True

            if (np.diagflat(gstate) == player).all():
                return True
            if (np.diagflat(gstate[::-1, :]) == player).all():
                return True

            return False


        if check_player(1):
            return 'player_won'
        if check_player(-1):
            return 'player_lost'

        if np.absolute(self._state).sum() == self.ngrid**2:
            return 'draw'

        return 'continue'

    def _opponent_move(self):
        #TODO: add proper opponent
        pos_moves = np.argwhere(self._state == 0).reshape(-1)
        if len(pos_moves)==1:
            move = pos_moves[0]
        else:
            move = np.random.choice(pos_moves)
        self._state[move] = -1


    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()


        grid_state = self._state[action]

        if grid_state == 0:
            self._state[action] = 1

        game_state = self._check_game_state()
        print(game_state)


        if game_state == 'player_won':
            r = 1
        elif game_state == 'draw':
            r = 0
        elif game_state == 'player_lost':
            raise KeyError('Player cannot lose after own turn!')
        elif game_state == 'continue':
            self._opponent_move()
            game_state = self._check_game_state()

            if game_state == 'draw':
                r = 0
            elif game_state == 'player_lost':
                r = -1
            elif game_state == 'player_won':
                raise KeyError('Cannot win after opponetn turn!')
            elif game_state == 'continue':
                r = 0
                #action_mask = self._state==0
                return ts.transition(np.array([self._state], dtype=np.int32), reward=r, discount=self.discount)
            else:
                raise KeyError('Inv key')
        else:
            raise KeyError('Inv key')

        self._episode_ended = True

        #action_mask = self._state==0
        return ts.termination(np.array([self._state], dtype=np.int32), r)

def render_text(arr, ngrid=3):

    print('BOARD:')
    board = arr.reshape((ngrid, ngrid))
    mapXO = {-1:'O', 1:'X', 0:' '}
    for i, row in enumerate(board):
        row = [mapXO[v] for v in row] #.astype(str)
        print(' | '.join(row))
        if i<ngrid-1:
            print('-'*len(row)*3)




class CardGameEnv(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,), dtype=np.int32, minimum=0, name='observation')
    self._state = 0
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = 0
    self._episode_ended = False
    return ts.restart(np.array([self._state], dtype=np.int32))

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    # Make sure episodes don't go on forever.
    if action == 1:
      self._episode_ended = True
    elif action == 0:
      new_card = np.random.randint(1, 11)
      self._state += new_card
    else:
      raise ValueError('`action` should be 0 or 1.')

    if self._episode_ended or self._state >= 21:
      reward = self._state - 21 if self._state <= 21 else -21
      return ts.termination(np.array([self._state], dtype=np.int32), reward)
    else:
      return ts.transition(
          np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)

def play_one(env, policy):

    time_step = env.reset()
    try:
        time_step.observation.board
        namedtuple_=True
    except AttributeError:
        namedtuple_=False

    ret = 0
    while True:

        if namedtuple_: # and (not time_step.is_last()):
            print('allowed moves:')
            print(time_step.observation.allowed_moves.numpy()) #.numpy()[0])
            board_arr = time_step.observation.board.numpy()[0]
        else:
            board_arr = time_step.observation.numpy()[0]


        render_text(board_arr)

        if time_step.is_last():
            break

        action = policy.action(time_step)
        print(action.action.numpy()[0])
        time_step = env.step(action)
        ret += time_step.reward.numpy()[0]
        print(ret)

    print('Ended:')
    print('RETURN:', ret)
    print()


if __name__=='__main__':
    environment = XOEnv()
    utils.validate_py_environment(environment, episodes=5)

    environment = XOEnv_masked()

    utils.validate_py_environment(environment, episodes=5)


    tf_env = tf_py_environment.TFPyEnvironment(environment)

    print('Checking conversion to tf env:')
    print(isinstance(tf_env, tf_environment.TFEnvironment))
    print("TimeStep Specs:", tf_env.time_step_spec())
    print("Action Specs:", tf_env.action_spec())
    print('Done.')

    print('Running dummy games:')
    place_X = np.array((0,0), dtype=np.int32)

    environment = XOEnv()
    time_step = environment.reset()
    print(time_step)
    cumulative_reward = time_step.reward
    ngames = 0
    stats = []
    for _ in range(30):
        place_X = np.random.randint(0, 9, size=1, dtype=np.int32)
        time_step = environment.step(place_X)
        if time_step.is_last():
            ngames += 1

            stats.append(time_step.reward)
            print(time_step, time_step.reward)
            print()
        cumulative_reward += time_step.reward

    print(time_step)
    print('Final Reward = ', cumulative_reward)
    print(ngames)
    print(np.unique(stats, return_counts=True))


