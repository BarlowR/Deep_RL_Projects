#https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
#http://incompleteideas.net/book/RLbook2020.pdf

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

import gym




class policy_estimator_network():

    #This is built on the OpenAi gym environment model 
    def __init__(self, envs):

        if type(envs) is list:
            self.environment = envs[0]
            self.environments = envs
        else:
            self.environment = envs
            self.environments = None


        self.action_space = np.arange(self.environment.action_space.n)
        self.n_inputs = self.environment.observation_space.shape[0]
        self.n_outputs = self.environment.action_space.n
        
        # This is just a standin network; specify your own when you initialize the class.
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16), 
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

    def run_episode(self, env):
        #print(id(self.network))
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        done = False
        while done == False:
            # Get actions and sample an action
            action_probs = self.predict(torch.FloatTensor(s_0)).detach().numpy()
            action = np.random.choice(self.action_space, p=action_probs)
            
            #print("\r \n ",len(states), "        step",  end="\033[F")

            # take a step in the environment
            s_1, r, done, _ = env.step(action)

            #print("\r \n ", len(states), "        post step", end="\033[F")
            #append items to our lists
            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

        return((states, actions, rewards))


    def reinforce(self, env = None, num_episodes=2000,
                  batch_size=5, gamma=0.99, learning_rate = 0.001,
                  batching_function = None):

        if env is None:
            env = self.environment
        # Set up lists to hold results
        total_rewards = []
        batch_rewards = []
        batch_actions = []
        batch_states = []
        batch_counter = 1
        
        # Define optimizer
        optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        ep = 0

        #run until we've trained on the defined number of episodes
        while ep < num_episodes:
            

            #run batch_number of episodes with the current network, and add all results into a batch list
            if batching_function is None:
                batches = []
                for _ in range(batch_size):
                    batches.append(self.run_episode(env))
            else: batches = batching_function(batch_size)

            #split the batches out into batch_rewards, batch_states and batch_actions
            for (states, actions, rewards) in batches:
                
                batch_rewards.extend(self.discount_rewards(rewards, gamma))

                batch_states.extend(states)
                
                batch_actions.extend(actions)
                               
                total_rewards.append(sum(rewards))
            
                
            #set gradients to 0
            optimizer.zero_grad()


            #create tensors of our batches
            state_tensor = torch.FloatTensor(batch_states)

            reward_tensor = torch.FloatTensor(batch_rewards)
            
            action_tensor = torch.LongTensor(batch_actions) #this needs to be a LongTensor since we'll be using it to select action probabilities
            

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


    def batch_multiprocess(self, batch_size):
        #this should only be called if you've passed in multiple environment instances as a list to the initialization function
        if self.environments is not None:
            batches = []
            for index in range(batch_size):
                env = self.environments[index % len(self.environments)]
                batches.append(self.run_episode(env))
            return batches
        else:
            raise Exception("policy network not initialized as multi-environment")



if __name__ == "__main__":

    gym.envs.register(
             id='Cappedlunar-v0',
             entry_point='gym.envs.box2d:LunarLander',
             max_episode_steps=400,
        )

    run_env = gym.make('Cappedlunar-v0')

    envs = [gym.make('Cappedlunar-v0')] * 10

    policy_estimator = policy_estimator_network(run_env)

    policy_estimator.reinforce(gamma = 0.999, learning_rate = 0.01, 
                            batching_function = policy_estimator.batch_multiprocess)
