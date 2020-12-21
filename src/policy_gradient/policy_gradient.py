#https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
#http://incompleteideas.net/book/RLbook2020.pdf

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
import gym
import time




class policy_estimator_network():

    #This is built on the OpenAi gym environment model 
    def __init__(self, envs):

        if type(envs) is list:
            self.environment = envs[0]
            self.environments = envs
            
            #self.mp_queue = mp.Queue()
        else:
            self.environment = envs
            self.environments = None


        self.action_space = np.arange(self.environment.action_space.n)
        self.n_inputs = self.environment.observation_space.shape[0]
        self.n_outputs = self.environment.action_space.n
        
        # This is just a standin network; specify your own when you initialize the class.
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 32), 
            nn.ReLU(), 
            nn.Linear(32, 16), 
            nn.ReLU(), 
            nn.Linear(16, self.n_outputs),
            nn.Softmax(dim=-1))

    def predict(self, state):
        #state needs to be a FloatTensor. returns a FloatTensor
        action_probs = self.network(state)
        return action_probs


    def discount_rewards(self, rewards, gamma=0.99):
        r = np.array([gamma**i * rewards[i] 
            for i in range(len(rewards))])
        # Reverse the array direction for cumsum and then
        # revert back to the original order
        r = r[::-1].cumsum()[::-1]
        return r

    def run_episode(self, env, queue = False, render = False):
        #print(id(self.network))
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        done = False
        while done == False:
            if render: env.render()
            # Get actions and sample an action
            action_probs = self.predict(torch.FloatTensor(s_0)).detach().numpy()
            action = np.random.choice(self.action_space, p=action_probs)
            
            # take a step in the environment
            s_1, r, done, _ = env.step(action)

            #print("\r \n ", len(states), "        post step", end="\033[F")
            #append items to our lists
            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

        if (queue):
            self.mp_queue.put((states, actions, rewards))
        return((states, actions, rewards))


    def reinforce(self, env = None, num_episodes=2000,
                  batch_size=5, gamma=0.99, learning_rate = 0.001):

        if env is None:
            env = self.environment

        # Set up lists to hold results
        total_rewards = []
        batch_rewards = []
        batch_actions = []
        batch_render = []
        batch_states = []
        batch_counter = 1
        
        # Define optimizer
        optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        ep = 0

        #run until we've trained on the defined number of episodes
        while ep < num_episodes:
            

            #run batch_number of episodes with the current network, and add all results into a batch list
            if self.environments is None:
                batches = []
                for _ in range(batch_size):
                    batches.append(self.run_episode(env))
            else:
                batch_size = len(self.environments)
                start = time.time()
                batches = self.batch_multiprocess()
                '''ulti_end = time.time()
                
                batches = []
                for _ in range(batch_size):
                    batches.append(self.run_episode(env))
                sequential_end = time.time()
                

                print("sequential:", sequential_end-multi_end, "batch:", multi_end-start)
                '''

            #split the batches out into batch_rewards, batch_states and batch_actions
            for (states, actions, rewards) in batches:

                

                #discount rewards
                rewards = self.discount_rewards(rewards, gamma)

                #print(rewards[1], rewards[-1])
                batch_rewards.extend(rewards)

                batch_states.extend(states)
                
                batch_actions.extend(actions)

                total_rewards.append(sum(rewards))
            #print("\n")                  
                
            #set gradients to 0
            optimizer.zero_grad()


            #create tensors of our batches
            state_tensor = torch.FloatTensor(batch_states)
            #print(len(state_tensor))

            reward_tensor = torch.FloatTensor(batch_rewards)
            
            action_tensor = torch.LongTensor(batch_actions) #this needs to be a LongTensor since we'll be using it to select action probabilities
            

            #scale our reward vector by substracting the mean from each element and scaling to unit variance by dividing by the standard deviation and machine epsilon.)
            reward_tensor = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + np.finfo(np.float32).eps)


            # Calculate the log probability of all possible actions at all states
            logprob = torch.log(self.predict(state_tensor))

            #our 
            action_tensor = torch.reshape(action_tensor, (len(action_tensor), 1))

            selected_logprobs = reward_tensor * torch.gather(logprob, 1, action_tensor).squeeze()

            loss = -selected_logprobs.mean()
            
            # Calculate gradients
            loss.backward()
            # Apply gradients
            optimizer.step()
            
                
            avg_rewards = np.mean(total_rewards[-100:])

            # Print running average
            print("\r \n", 
                "                                      \n",
                 "Episode: ", ep, "Avg Reward: ", str(avg_rewards)[0:6],
                  end="\033[F \033[F")
            ep += batch_size

            #reset our batch lists
            batch_rewards = []
            batch_actions = []
            batch_states = []
                    
        return total_rewards


    def batch_multiprocess(self):
        #this will be called if you've passed in multiple environment instances as a list to the initialization function
        if self.environments is not None:

            batches = []

            with mp.get_context("spawn").Pool(processes=4) as p:
                batches = p.map(self.run_episode, self.environments)
            #batches = [self.run_episode(self.environments[i]) for i in range (4)]
            return batches
        else:
            raise Exception("policy network not initialized as multi-environment")



class image_policy_estimator_network(policy_estimator_network):

    def __init__(self, envs):
        super(image_policy_estimator_network, self).__init__(envs)

        self.network = nn.Sequential(
                
                nn.Conv2d(
                in_channels=4,
                out_channels=16,
                kernel_size=8,
                stride=4,
                padding=2),

                nn.ReLU(),

                nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1),

                nn.ReLU(),

                nn.Flatten(),

                nn.Linear(
                in_features=3200,
                out_features=self.n_outputs),

                nn.Softmax(dim = -1)
                )

    def predict(self, state):
        #state needs to be a FloatTensor. returns a FloatTensor
        if state.dim() == 3:
            state = state.unsqueeze(dim=0)
            action_probs = self.network(state)
            action_probs = action_probs.squeeze(dim=0)
        else:
            action_probs = self.network(state)

        return action_probs


'''
if __name__ == "__main__":


    gym.envs.register(
             id='Cappedlunar-v0',
             entry_point='gym.envs.box2d:LunarLander',
             max_episode_steps=800,
        )

    run_env = gym.make('Cappedlunar-v0')

    envs = []
    for i in range(4):
        envs.append(gym.make('Cappedlunar-v0'))

    policy_estimator = policy_estimator_network(envs)
    
    policy_estimator.reinforce(num_episodes = 30000, gamma = 0.999, batch_size = 5, learning_rate = 0.01)'''