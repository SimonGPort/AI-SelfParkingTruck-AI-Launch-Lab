""" Python file for DDPG """
import random
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Input
from variables import *
import sys
import gym


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=NOISE_THETA, dt=NOISE_DT, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        # self.x_prev = 0
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1][0]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + GAMMA * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `TAU`, which is much less than one.
@tf.function
def update_target(target_weights, weights, TAU):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * TAU + a * (1 - TAU))


def get_actor(num_states, num_actions, upper_bound):
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=KERNEL_INITIALIZATION_MINVAL, maxval=KERNEL_INITIALIZATION_MAXVAL)
    inputs = Input(shape=(num_states,))
    out = inputs
    for hidden_layer in HIDDEN_LAYERS_ACTOR:
        out = Dense(hidden_layer, activation=ACTOR_ACTIVATION)(out)

    outputs = Dense(num_actions, activation=OUTPUT_ACTIVATION, kernel_initializer=last_init)(out)
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(num_states, num_actions, upper_bound):
    # State as input
    state_input = Input(shape=(num_states))
    state_out = state_input
    for state_hidden in HIDDEN_LAYERS_CRITIC[0]:
        state_out = Dense(state_hidden, activation=CRITIC_ACTIVATION)(state_out)

    # Action as input
    action_input = Input(shape=(num_actions))
    action_out = action_input
    for action_hidden in HIDDEN_LAYERS_CRITIC[1]:
        action_out = Dense(action_hidden, activation=CRITIC_ACTIVATION)(action_out)

    # Both are passed through seperate layer before concatenating
    concat = Concatenate()([state_out, action_out])
    out = concat
    for hidden_critic in HIDDEN_LAYERS_CRITIC[2]:
        out = Dense(hidden_critic, activation=CRITIC_ACTIVATION)(out)

    #Changed the number of ouputs to 1, because critic only outputs a Q value.
    outputs = Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    #print(f"noise: {[n for n in noise]}")
    # noise = np.array([random.uniform(-0.45, 0.45)])
    #print(f"Sampled actions: {[round(sampled_action, 4) for sampled_action in sampled_actions.numpy()]} "
    #      f"| Noise: {[round(n, 4) for n in noise]}")
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise
    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


def explore(noise_object):
    noise = noise_object()
    legal_action = np.clip(noise, lower_bound, upper_bound)
    return [np.squeeze(legal_action)]


def main():
    buffer = Buffer(BUFFER_CAPACITY, BATCH_SIZE)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    for ep in range(TOTAL_EPISODES):

        if not ep % SHOW_EVERY:
            env.render(True)
        else:
            env.render(False)


        prev_state = env.reset()
        #env.render()
        episodic_reward = 0


        while True:
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            # if ep < 100:
            #     action = explore(ou_noise_train)
            # else:
            #     action = policy(tf_prev_state, ou_noise_run)
            action = policy(tf_prev_state, ou_noise_run)
            # Receive state and reward from environment.
            state, reward, done, info = env.step(action)

            buffer.record((prev_state, action, reward, state))
            episodic_reward += reward

            buffer.learn()
            update_target(target_actor.variables, actor_model.variables, TAU)
            update_target(target_critic.variables, critic_model.variables, TAU)

            # End this episode when `done` is True
            if done:
                break

            prev_state = state

        ep_reward_list.append(episodic_reward)
        print("********")
        print("Episode " + str(ep) + " Completed with Reward: " + str(round(episodic_reward)) + "!")
        print("Last Action was: " + str(np.around(action[0])))
        print("Last Obs was: " + str(np.around(state, 4)))

        if ep % AGGREGATE_STATS_EVERY == 0:

            # Plotting graph
            # Episodes versus Avg. Rewards
            num_lst = np.arange(len(ep_reward_list))
            plt.scatter([round(num, 0) for num in num_lst], ep_reward_list, c='y', alpha=0.7, marker='+')
            plt.plot(avg_reward_list, c='b')

            plt.xlabel("Episode", weight='bold')
            plt.ylabel("Reward", weight='bold')
            plt.title('Graph of rewards over episodes')
            plt.savefig(f'{MODEL_NAME}_Episode_Rewards')

        average_reward = sum(ep_reward_list[-AGGREGATE_STATS_EVERY:]) / AGGREGATE_STATS_EVERY
        min_reward = min(ep_reward_list[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_reward_list[-AGGREGATE_STATS_EVERY:])
        # Save model every so often
        if ((ep % SAVE_MODEL_PER) == 0) or ((episodic_reward > max(ep_reward_list)) and (len(ep_reward_list) > 20)):
            actor_model.save(
                f'models/ep{ep}_{average_reward:.0f}avg/Actor')
            critic_model.save(
                f'models/ep{ep}_{average_reward:.0f}avg/Critic')

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-NUM_EPISODE_MEAN:])
        #print(f"Ep{ep} Reward: {episodic_reward:.1f} Avg Reward (last {NUM_EPISODE_MEAN:.1f} eps): {avg_reward:.1f}")
        avg_reward_list.append(avg_reward)


# Learning from a previously trained model
# if __name__ == "__main__":
#     env = Environment()
#     #env = gym.make("Pendulum-v0")
#     # num_states = env.observation_space
#     num_states = env.observation_space.shape[0]
#     # num_actions = env.action_space[2][0]
#     num_actions = env.action_space.shape[0]
#     # upper_bound = env.action_space[1]
#     # lower_bound = env.action_space[0]
#     upper_bound = env.action_space.high[0]
#     lower_bound = env.action_space.low[0]
#
#     # ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(NOISE_STD) * np.ones(1))
#     ou_noise = 0
#     actor_model = get_actor(num_states, num_actions, upper_bound)
#     critic_model = get_critic(num_states, num_actions, upper_bound)
#
#     ep = 300
#     avg_reward = -82
#     actor_model.load_weights(f'models/ep{ep}_{avg_reward:.0f}avg/Actor')
#     critic_model.load_weights(f'models/ep{ep}_{avg_reward:.0f}avg/Critic')
#
#     target_actor = get_actor(num_states, num_actions, upper_bound)
#     target_critic = get_critic(num_states, num_actions, upper_bound)
#
#     # Making the weights equal initially
#     target_actor.set_weights(actor_model.get_weights())
#     target_critic.set_weights(critic_model.get_weights())
#
#     critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LR)
#     actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LR)
#     main()

# Learning from scratch. Comment out the previous if __name__ part if you want to use this
if __name__ == "__main__":
    env = Environment()
    # env = gym.make("Pendulum-v0")
    # num_states = env.observation_space
    num_states = env.observation_space.shape[0]
    # num_actions = env.action_space[2][0]
    num_actions = env.action_space.shape[0]
    # upper_bound = env.action_space[1]
    # lower_bound = env.action_space[0]
    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    ou_noise_run = OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(NOISE_STD) * np.ones(num_actions))
    ou_noise_train = OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(BIG_NOISE_STD) * np.ones(num_actions))

    actor_model = get_actor(num_states, num_actions, upper_bound)
    critic_model = get_critic(num_states, num_actions, upper_bound)

    target_actor = get_actor(num_states, num_actions, upper_bound)
    target_critic = get_critic(num_states, num_actions, upper_bound)

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LR)
    actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LR)
    main()
