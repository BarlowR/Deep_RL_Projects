import policy_gradient as pg 
import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time
import pickle
import torch.nn as nn




headless = True
learning_rate = 0.0001
batch_size = 10


if __name__ == "__main__":




    initial_time = time.time()


    run_env = gym.make('Breakout-v0')
    run_env = gym.wrappers.Monitor(run_env, "./recordings", video_callable=lambda episode_id: (episode_id%100 == 0), force=True)
    run_env = pg.ImagePreProcessingWrapper(run_env, 4)


    envs = []
    for _ in range(10):
        env = gym.make('Breakout-v0')
        envs.append(pg.ImagePreProcessingWrapper(env, 4))


    policy_gradient_agent = pg.image_policy_estimator_network(envs)

    
    rewards_over_time = []
    time_per_comp = []

    for i in range(1000):


        if i%10 == 0 and not headless: policy_gradient_agent.run_episode(run_env, render = True)
        

        time_per_comp.append((time.time()-initial_time)/60)

        if not headless:
            plt.subplot(2,2,1)
            plt.plot(rewards_over_time)
            plt.ylabel("Reward")
            plt.subplot(2,2,2)
            plt.plot(time_per_comp)
            plt.xlabel("Episodes (x100)")
            plt.ylabel("Time (minutes, cumulative)")
            plt.subplot(2,2,3)
            plt.imshow(policy_gradient_agent.network[0].weight.data.numpy()[0][1,:,:])
            plt.subplot(2,2,4)
            plt.imshow(policy_gradient_agent.network[0].weight.data.numpy()[0][1,:,:])
            plt.pause(0.0001)

        rewards = policy_gradient_agent.reinforce(run_env, num_episodes = 100, batch_size = batch_size, learning_rate = learning_rate)
        rewards_over_time.append(sum(rewards)/100)

        pickle.dump((policy_gradient_agent.network, rewards_over_time, time_per_comp), open( "atari_policy.p", "wb" ) )

        
