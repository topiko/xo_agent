import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.utils import common
from xo_env import XOEnv
from utils import compute_avg_return

from tf_agents.utils import common
from tf_agents.environments import tf_py_environment
from tf_agents.policies import q_policy
from tf_agents.networks import q_network
from tf_agents.policies import greedy_policy, epsilon_greedy_policy, random_tf_policy
from tf_agents.trajectories import time_step as ts

from tf_agents.specs import array_spec, tensor_spec
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from tensorflow.keras.initializers import RandomNormal

from xo_env import CardGameEnv, XOEnv_masked
from xo_env import play_one

XO = 'O'
# Init the env and cast it to tf env.
#env = XOEnv() #XOEnv_masked() #CardGameEnv() #
env = XOEnv_masked(player=XO)

tf_env = tf_py_environment.TFPyEnvironment(env)

def observation_and_action_constraint_splitter(observation):
    #if isinstance(observation.board, tf.Tensor):
    #    return observation.board, observation.allowed_moves[0]
    return observation.board, observation.allowed_moves

if isinstance(env, XOEnv):
    input_tensor_spec = tf_env.observation_spec()
    observation_and_action_constraint_splitter_fun = None
else:
    input_tensor_spec = tf_env.observation_spec().board
    observation_and_action_constraint_splitter_fun = observation_and_action_constraint_splitter

initializer = RandomNormal(mean=.0, stddev=.1)
FC_LAYER_PARAMS = (9*5, 9*4, 9*2, )




# Define the qnet, policy, and agent.
qnet = q_network.QNetwork(
    input_tensor_spec=input_tensor_spec,  #.board, #env.observation_spec(),
    action_spec=tf_env.action_spec(),
    fc_layer_params=FC_LAYER_PARAMS,
    kernel_initializer=initializer,
    activation_fn=tf.keras.activations.elu)

global_step = tf.compat.v1.train.get_or_create_global_step()

NSTEPUPDATE = 2
B_TEMP = 50
agent = dqn_agent.DqnAgent(
    time_step_spec=tf_env.time_step_spec(),
    action_spec=tf_env.action_spec(),
    epsilon_greedy=None,
    boltzmann_temperature=B_TEMP,
    observation_and_action_constraint_splitter=observation_and_action_constraint_splitter_fun,
    q_network=qnet,
    n_step_update=NSTEPUPDATE,
    optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
    train_step_counter=global_step)


print('Playing DQN.')
play_one(tf_env, agent.policy)

# Make a replay buffer.
replay_buffer_capacity = 200

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=replay_buffer_capacity)


# Add things to observe/collect.
num_episodes = tf_metrics.NumberOfEpisodes()
env_steps = tf_metrics.EnvironmentSteps()
avg_return = tf_metrics.AverageReturnMultiMetric(
    reward_spec=env.reward_spec())
add_to_buffer = replay_buffer.add_batch
observers = [num_episodes, env_steps, avg_return, add_to_buffer]


# Driver drives the whole system.
N_EP = 50
driver = dynamic_episode_driver.DynamicEpisodeDriver(
    env=tf_env,
    policy=agent.collect_policy,
    observers=observers,
    num_episodes=N_EP)

train_checkpointer = common.Checkpointer(
    ckpt_dir='checkpoint_{}/'.format(XO),
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)


# Initial driver.run will reset the environment and initialize the policy.
final_time_step, policy_state = driver.run(num_episodes=1)

print('final_time_step', final_time_step)
print('Number of Steps: ', env_steps.result().numpy())
print('Number of Episodes: ', num_episodes.result().numpy())


for _ in range(5):
    print('='*20)
    play_one(tf_env, agent.collect_policy)


# Read the replay buffer as a Dataset,
# read batches of 4 elements, each with 2 timesteps:
dataset = replay_buffer.as_dataset(
    sample_batch_size=100,
    num_steps=NSTEPUPDATE+1)

iterator = iter(dataset)

agent.train = common.function(agent.train)

def show_policy(policy):

    time_step_spec = tf_env.time_step_spec()
    batch_size = 1
    for i in range(9):
        observation = tf.ones([batch_size] + time_step_spec.observation.shape.as_list(), dtype=np.int32)*i
        time_steps = ts.restart(observation, batch_size=batch_size)
        print(time_steps.observation.numpy(), '-->', policy.action(time_steps).action.numpy())
        print(qnet.call(time_steps.observation))

def train_one_iteration(show_stats=False):

    # Collect a few steps using collect_policy and save to the replay buffer.
    driver.run()

    # Sample a batch of data from the buffer and update the agent's network.
    #for _ in range(100):
    experience, unused_info = next(iterator)

    train_loss = agent.train(experience)


    avg_return=None
    if show_stats:
        iteration = agent.train_step_counter.numpy()
        print ('iteration: {0}'.format(iteration))
        print('Number of Steps:'.rjust(25), env_steps.result().numpy())
        print('Number of Episodes:'.rjust(25), num_episodes.result().numpy())
        nep = 100
        fwins, flosses, fdraws = compute_avg_return(tf_env, agent.policy, num_episodes=nep)
        print('nwins={:d}, ndraws={:d}, nlosses={:d}'.format(
            int(fwins*nep),
            int(fdraws*nep),
            int(flosses*nep)).rjust(25))

        #show_policy(agent.policy)


    if show_stats:
        return num_episodes.result().numpy(), train_loss.loss, fwins, flosses, fdraws


def run_once(data, Niter=1000, report_interval=10, plot_=False):

    data_ = np.zeros((Niter//report_interval, 5))


    num_episodes.reset()
    env_steps.reset()

    for i in range(Niter):
        if (i+1)%report_interval==0:
            data_[i//report_interval, :] = train_one_iteration(show_stats=True)
            train_checkpointer.save(global_step)
        else:
            train_one_iteration()


    if data is not None:
        data_[:, 0] += data[:, 0].max() #+ N_EP*(report_interval-1)
        data = np.concatenate((data, data_))
    else:
        data = data_

    if plot_:
        plot_runs(data)
    return data

def plot_runs(data):
    _, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    ax1.plot(data[:, 0], data[:, 1], '-o', lw=1, ms=3, label='loss')
    ax2.plot(data[:, 0], data[:, 2], '-o', lw=1, ms=3, label='frac_wins')
    ax2.plot(data[:, 0], data[:, 3], '-o', lw=1, ms=3, label='frac_losses')
    ax2.plot(data[:, 0], data[:, 4], '-o', lw=1, ms=3, label='frac_draws')
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    ax2.set_xlabel('# games')
    plt.show()



try:
    data = np.load('checkpoint_{XO}/evolution_{XO}.npy'.format(XO=XO))
except FileNotFoundError:
    data = None

plot_ = True
niter = 40
while True:
    data = run_once(data, Niter=niter, report_interval=20, plot_=plot_)
    play_one(tf_env, agent.policy)
    np.save('checkpoint_{XO}/evolution_{XO}.npy'.format(XO=XO), data)
    niter = int(input('Next niter='))

