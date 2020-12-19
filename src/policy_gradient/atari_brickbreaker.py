import policy_gradient as pg 
import gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import scipy
import time
import pickle



class AtariPreProcessingWrapper(gym.Wrapper):
    #initialize the wrapper
    def __init__(self, env, queue_length):
        super().__init__(env)
        self.env = env
        self.stack = deque()
        self.queue_length = queue_length

    #process the frames from 210 × 160 pixel images with a 128 color palette down to 84x84 downscaled cropped greyscale image
    def process (self,image_state):
        grey = np.dot(image_state[...,:3], [0.299, 0.587, 0.114])
        grey_downsampled = grey[::2,::2]
        grey_cropped = grey_downsampled[18:98]
        return grey_cropped

    def reset(self):
        #reset the environment and add the state as the first in our queue
        
        self.stack.clear()
        self.stack.append(self.process(self.env.reset()))

        #take populate our state queue by taking steps with an arbitrary action 
        for _ in range(self.queue_length-1):
            next_state, reward, done, info, = self.env.step(1)
            self.stack.append(self.process(next_state))

        return np.array(self.stack)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        self.stack.append(self.process(next_state))
        self.stack.popleft()
        return (np.array(self.stack), reward, done, info)



initial_time = time.time()


run_env = gym.make('Breakout-v0')
run_env = gym.wrappers.Monitor(run_env, "recordings", video_callable=lambda episode_id: (episode_id%50 == 0), force=True)
run_env = AtariPreProcessingWrapper(run_env, 4)

policy_gradient_agent = pg.policy_estimator_network_atari(run_env)

plt.show()  
rewards_over_time = []
time_per_comp = []

for i in range(1000):


    s_0 = run_env.reset()
    action_space = np.arange(run_env.action_space.n)
    done = False

    while not done:
        run_env.render()
        action_probs = policy_gradient_agent.predict(s_0).detach().numpy()
        action = np.random.choice(action_space, p=action_probs)
        s_1, r, done, _ = run_env.step(action)
        time.sleep(0.01)
        #print("\r", r)
        s_0 = s_1
    

    time_per_comp.append((time.time()-initial_time)/60)

    plt.subplot(2,1,1)
    plt.plot(rewards_over_time)
    plt.ylabel("Reward")
    plt.subplot(2,1,2)
    plt.plot(time_per_comp)
    plt.xlabel("Episodes (x100)")
    plt.ylabel("Time (minutes, cumulative)")
    plt.pause(0.0001)

    rewards = policy_gradient_agent.reinforce(run_env, num_episodes = 100)
    rewards_over_time.append(sum(rewards)/100)

    pickle.dump(policy_gradient_agent.network, open( "policy.p", "wb" ) )

    
plt.show()   