from variables import *
from environment import Environment
from ddpg import *


def policy(state):
    sampled_actions = tf.squeeze(actor_model(state))
    sampled_actions = sampled_actions.numpy()
    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

def main_test():
    state = env.reset()
    env.render(mode='human')
    episodic_reward = 0

    while True:
        env.render(mode='human')
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)

        action = policy(state)

        # Receive state and reward from environment.
        state, reward, done, info = env.step(action)
        episodic_reward += reward

        # End this episode when `done` is True
        if done:
            break

    print(" Completed with Reward: " + str(round(episodic_reward)) + "!")
    print(action)


# doesn't work
if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
    # num_states = env.observation_space
    num_states = env.observation_space.shape[0]
    # num_actions = env.action_space[2][0]
    num_actions = env.action_space.shape[0]
    # upper_bound = env.action_space[1]
    # lower_bound = env.action_space[0]
    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    actor_model = get_actor(num_states, num_actions, upper_bound)

    ep = 300
    avg_reward = -82
    actor_model.load_weights(f'models/ep{ep}_{avg_reward:.0f}avg/Actor')

    main_test()

