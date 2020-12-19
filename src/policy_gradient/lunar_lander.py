import policy_gradient as pg 
import gym
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle


gym.envs.register(
         id='Cappedlunar-v0',
         entry_point='gym.envs.box2d:LunarLander',
         max_episode_steps=400,
    )

run_env = gym.make('Cappedlunar-v0')
run_env = gym.wrappers.Monitor(run_env, "recordings", video_callable=lambda episode_id: False, force=True)

initial_time = time.time()

#policy_est = pickle.load( open( "policy.p", "rb" ) )
policy_est = pg.policy_estimator_network(envs[0])
    
rewards_over_time = []
time_per_comp = []

for i in range(1000):

	rewards = policy_est.reinforce(run_env, num_episodes = 100)
	rewards_over_time.append(sum(rewards)/100)

	time_per_comp.append((time.time()-initial_time)/60)

	plt.subplot(2,1,1)
	plt.plot(rewards_over_time)
	plt.ylabel("Reward")
	plt.subplot(2,1,2)
	plt.plot(time_per_comp)
	plt.xlabel("Episodes (x100)")
	plt.ylabel("Time (minutes, cumulative)")
	plt.pause(0.0001)

	#pickle.dump( policy_est, open( "policy.p", "wb" ) )

	if i%10 == 0:
		s_0 = run_env.reset()
		action_space = np.arange(run_env.action_space.n)
		done = False

		for _ in range(400):
		    run_env.render()
		    if not done:
		        action_probs = policy_est.predict(s_0).detach().numpy()
		        action = np.random.choice(action_space, p=action_probs)
		        s_1, r, done, _ = run_env.step(action)
		        #print("\r", r)
		        s_0 = s_1
		    else: break
plt.show()   

env.close()