import numpy as np

def compute_avg_return(environment, policy, num_episodes=10):

    total_return = np.zeros(num_episodes)
    for i in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return[i] = episode_return.numpy()

    fwins = (total_return==1).sum()/num_episodes
    flosses = (total_return==-1).sum()/num_episodes
    fdraws = (total_return==0).sum()/num_episodes

    return fwins, flosses, fdraws
    #return avg_return.numpy()[0]

