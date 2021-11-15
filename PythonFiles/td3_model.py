from typing import List
import tensorflow as tf
import tensorflow.keras.losses as loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Concatenate, Input
from variables import *
from keras.models import load_model


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class Agent:
    def __init__(self, alpha, beta, input_dims, tau, env,
                 gamma=GAMMA, update_actor_interval=UPDATE_ACTOR_INTERVAL, warmup=5000,
                 n_actions=2, max_size=MAX_MEM_SIZE, batch_size=100, noise=0.2, load_models_from_file=None):
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval
        self.create_models()
        self.compile_models()
        # Loads and compiles a model from file location. (Not working yet)
        if load_models_from_file is not None:
            self.load_models(load_models_from_file)
        self.noise = noise
        self.update_network_parameters()

    @staticmethod
    def get_critic(n_states: float, n_actions: float, upper_bound: float,
                   critic_hidden_layers: List[float]) -> tf.keras.Model:
        """
        This function creates a critic model.
        @param n_states: The number of states taken as input for the model.
        @param n_actions: The number of actions taken as input for the model.
        @param upper_bound: The upper bound value of actions.
        @param critic_hidden_layers: A list of layers with their respective number of neurons.
        @return: Tensorflow model.
        """
        state_input = Input(shape=n_states)
        action_input = Input(shape=n_actions)
        inputs = [state_input, action_input]

        out = Concatenate()(inputs)
        for hidden_layer in critic_hidden_layers:
            out = Dense(hidden_layer, activation=TD3_CRITIC_HIDDEN_ACTIVATION)(out)

        q = Dense(1, activation=None)(out)
        model = tf.keras.Model(inputs, q)
        return model

    @staticmethod
    def get_actor(n_states: float, n_actions: float, upper_bound: float,
                  actor_hidden_l: List[float]) -> tf.keras.Model:
        """
        This function creates an actor model.
        @param n_states: The number of states taken as input for the model.
        @param n_actions: The number of actions that the model returns.
        @param upper_bound: The upper bound value of actions.
        @param actor_hidden_l: A list of layers with their respective number of neurons.
        @return: Tensorflow model.
        """
        last_init = tf.random_uniform_initializer(minval=KERNEL_INITIALIZATION_MINVAL,
                                                  maxval=KERNEL_INITIALIZATION_MAXVAL)
        inputs = Input(shape=n_states)
        out = inputs
        for hidden_layer in actor_hidden_l:
            out = Dense(hidden_layer, activation=TD3_ACTOR_HIDDEN_ACTIVATION)(out)

        outputs = Dense(n_actions, activation=TD3_ACTOR_OUTPUT_ACTIVATION, kernel_initializer=last_init)(out)
        outputs = outputs * upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def save_models(self, main_folder, ep, avg_score):
        self.actor.save(f"{main_folder}/models/ep{ep}_avg_rew{avg_score:.0f}/Actor/")
        self.critic_1.save(f"{main_folder}/models/ep{ep}_avg_rew{avg_score:.0f}/Critic1/")
        self.critic_2.save(f"{main_folder}/models/ep{ep}_avg_rew{avg_score:.0f}/Critic2/")
        self.target_actor.save(f"{main_folder}/models/ep{ep}_avg_rew{avg_score:.0f}/Target_Actor/")
        self.target_critic_1.save(f"{main_folder}/models/ep{ep}_avg_rew{avg_score:.0f}/Target_Critic1/")
        self.target_critic_2.save(f"{main_folder}/models/ep{ep}_avg_rew{avg_score:.0f}/Target_Critic2/")

    def create_models(self):
        self.actor = self.get_actor(self.input_dims, self.n_actions, self.max_action, ACTOR_HIDDEN_LAYERS)
        self.target_actor = self.get_actor(self.input_dims, self.n_actions, self.max_action, ACTOR_HIDDEN_LAYERS)
        self.critic_1 = self.get_critic(self.input_dims, self.n_actions, self.max_action, CRITIC_1_HIDDEN_LAYERS)
        self.target_critic_1 = self.get_critic(self.input_dims, self.n_actions, self.max_action, CRITIC_1_HIDDEN_LAYERS)
        self.critic_2 = self.get_critic(self.input_dims, self.n_actions, self.max_action, CRITIC_2_HIDDEN_LAYERS)
        self.target_critic_2 = self.get_critic(self.input_dims, self.n_actions, self.max_action, CRITIC_2_HIDDEN_LAYERS)

    def load_models(self, load_models_from_file):
        self.actor = load_model(f'{load_models_from_file}/Actor/')
        self.target_actor = load_model(f'{load_models_from_file}/Target_Actor/')
        self.critic1 = load_model(f'{load_models_from_file}/Critic1/')
        self.target_critic_1 = load_model(f'{load_models_from_file}/Target_Critic1/')
        self.critic_2 = load_model(f'{load_models_from_file}/Critic2/')
        self.target_critic_2 = load_model(f'{load_models_from_file}/Target_Critic2/')

    def compile_models(self):
        self.actor.compile(optimizer=Adam(learning_rate=self.alpha), loss='mean')
        self.target_actor.compile(optimizer=Adam(learning_rate=self.alpha), loss='mean')
        self.critic_1.compile(optimizer=Adam(learning_rate=self.beta), loss='mean_squared_error')
        self.target_critic_1.compile(optimizer=Adam(learning_rate=self.beta), loss='mean_squared_error')
        self.critic_2.compile(optimizer=Adam(learning_rate=self.beta), loss='mean_squared_error')
        self.target_critic_2.compile(optimizer=Adam(learning_rate=self.beta), loss='mean_squared_error')

    def choose_action(self, observation: np.array) -> tf.constant:
        """
        This function takes actions based on the observations. During the warmup period, the actions are random,
        and then it's the actor model that makes them. Noise is added to the actions of the actor mode, which
        decays over time.
        @param observation: A numpy array describing the state of the Vortex simulation.
        @return: A tensorflow tensor of the actions taken by the model.
        """
        if self.time_step < self.warmup:
            mu = np.random.normal(scale=self.noise, size=(self.n_actions,))
        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            mu = self.actor(state)[0]  # returns a batch size of 1, want a scalar array

        # Apply epsilon greedy
        epsilon = MAX_EPSILON * np.exp(-self.time_step/STEPS_FOR_MIN_EPSILON)

        # Add noise to action
        self.noisy = np.random.uniform(low=-self.noise, high=self.noise, size=(self.n_actions,)) * epsilon
        mu_prime = mu + self.noisy

        mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)
        self.time_step += 1

        return mu_prime

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        """
        The function that allows the actor and critic to learn by updating the weights of their neurons.
        """
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(states_, training=True)
            # TODO: Ask what this means
            target_actions = target_actions + tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)

            target_actions = tf.clip_by_value(target_actions, self.min_action, self.max_action)

            q1_ = self.target_critic_1([states_, target_actions], training=True)
            q2_ = self.target_critic_2([states_, target_actions], training=True)

            q1 = tf.squeeze(self.critic_1([states, actions], training=True), 1)
            q2 = tf.squeeze(self.critic_2([states, actions], training=True), 1)

            # shape is [batch_size, 1], want to collapse to [batch_size]
            q1_ = tf.squeeze(q1_, 1)
            q2_ = tf.squeeze(q2_, 1)

            critic_value_ = tf.math.minimum(q1_, q2_)
            # in tf2 only integer scalar arrays can be used as indices
            # and eager execution doesn't support assignment, so we can't do
            # q1_[dones] = 0.0
            target = rewards + self.gamma * critic_value_ * (1 - dones)
            # critic_1_loss = tf.math.reduce_mean(tf.math.square(target - q1))
            # critic_2_loss = tf.math.reduce_mean(tf.math.square(target - q2))
            critic_1_loss = loss.MSE(target, q1)
            critic_2_loss = loss.MSE(target, q2)

        critic_1_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        with tf.GradientTape() as tape:
            new_actions = self.actor(states, training=True)
            critic_1_value = self.critic_1([states, new_actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_1_value)

        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

    def update_network_parameters(self, tau: float = None) -> None:
        """
        This function updates the network parameters with the value tau, which is very small. This stabilizes the
        learning process.
        """
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))

        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))

        self.target_critic_1.set_weights(weights)

        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))

        self.target_critic_2.set_weights(weights)
