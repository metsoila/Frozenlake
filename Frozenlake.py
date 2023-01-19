import gym
import numpy as np
import matplotlib.pyplot as plt



max_episodes = 100
max_steps = 100
gamma = 0.9
alpha = 0.5



def choose_action(qtable, state):
    """
    Chooses where to move either randomly or the most optimal if possible
    """

    #If state has optimal move, it will choose the best one
    if (np.max(qtable[state] > 0)):
        action = np.argmax(qtable[state])
    
    #Else it will take a random action
    # 0:Left 1:Down 2: Right, 3: Up
    else:
        action = np.random.randint(0,4) 

    return action


def evaluation(env, qtable):
    """
    Makes new runs every 10 episodes with updated Q-Table. Calculates
    average reward. 
    """
    rewards = [] 
    for i in range(max_episodes):
        
        state = env.reset()
        done = False
        total_rewards = 0

        for step in range(max_steps):
            action = np.argmax(qtable[state])
            new_state, reward, done, info = env.step(action)
            total_rewards += reward

            if done:
                rewards.append(total_rewards)
                break

            state = new_state
    
    env.close()
    avg_reward = sum(rewards)/max_episodes
    return avg_reward



def Q_algo(bool_slippery, bool_naive):
    """
    Sets the Q-Table and updates it. Algorithm calculates
    new values for matrix, so optimal steps can be found.

    param1: boolean whether field is slippery
    param1: boolean whether algorithm is naive learning or not
    """

    env = gym.make("FrozenLake-v1", is_slippery=bool_slippery)

    #Set empty Q-Table
    qtable = np.zeros((env.observation_space.n, env.action_space.n))

    rewards = []
    episodes = []


    for episode in range(max_episodes):

        done = False

        state = env.reset()


        while not done:
            action = choose_action(qtable, state)
            new_state, reward, done, info = env.step(action)

            #Update qtable, and if naive = False, we use the rule update
            if bool_naive:
                qtable[state, action] = reward+gamma*np.max(qtable[new_state])
            else:
                qtable[state, action] = qtable[state, action] + alpha*(reward+gamma*np.max(qtable[new_state]- qtable[state, action] ))

            state = new_state


        if episode%10 == 0:
            rewards.append(evaluation(env, qtable))
            episodes.append(episode)

    env.close()
    return rewards, episodes



def main():
    """
    Draws graphs for different tests
    """

    fig, axs = plt.subplots(3, 1)

    for i in range(10):
        #a) 
        rewards, episodes = Q_algo(False, True) #Not slippery, naive learning
        axs[0].plot(episodes, rewards, color='r')
        axs[0].set_title(f"a)")
        axs[0].grid(visible=True, axis="y")

        #b)
        rewards, episodes = Q_algo(True, True) #Slippery, not naive learning
        axs[1].plot(episodes, rewards, color='g')
        axs[1].set_title(f"b)")
        axs[1].grid(visible=True, axis="y")

        #c)
        rewards, episodes = Q_algo(True, False) #Slippery, naive learning
        axs[2].plot(episodes, rewards, color='b')
        axs[2].set_title(f"c)")
        axs[2].grid(visible=True, axis="y")
    
    plt.show()

if __name__ == "__main__":
    main()