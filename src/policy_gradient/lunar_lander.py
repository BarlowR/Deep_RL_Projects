import policy_gradient as pg 
import gym
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle



batch_size = 12
plot_frequency = 100
learning_rate = 0.001
gamma = 0.999




if __name__ == "__main__":
	gym.envs.register(
	         id='Cappedlunar-v0',
	         entry_point='gym.envs.box2d:LunarLander',
	         max_episode_steps=800,
	    )

	run_env = gym.make('Cappedlunar-v0')
	run_env = gym.wrappers.Monitor(run_env, "recordings", video_callable=lambda episode_id: True, force=True)

	envs = []
	for i in range(batch_size):
	        envs.append(gym.make('Cappedlunar-v0'))

	initial_time = time.time()

	#policy_est = pickle.load( open( "policy.p", "rb" ) )
	policy_est = pg.policy_estimator_network(envs)
	    
	rewards_over_time = []
	time_per_comp = []

	for i in range(1000):

		rewards = policy_est.reinforce(num_episodes = plot_frequency, gamma=0.999, learning_rate = learning_rate)
		rewards_over_time.append(sum(rewards)/plot_frequency)

		time_per_comp.append((time.time()-initial_time)/60)

		title = "Batch Size: " + str(batch_size) + ", Learning Rate: " + str(learning_rate) + ", Gamma: " + str(gamma)
		
		plt.subplot(2,1,1)
		plt.title(title)
		plt.plot(rewards_over_time)
		plt.ylabel("Reward")
		plt.subplot(2,1,2)
		plt.plot(time_per_comp)
		plt.xlabel("Episodes (x100)")
		plt.ylabel("Time (minutes, cumulative)")
		plt.pause(0.0001)

		#pickle.dump( policy_est, open( "policy.p", "wb" ) )

		if i%10 == 0:
			policy_est.run_episode(run_env, render = True)
	plt.show()   

