from environment import Environment
from td3_model import Agent
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from variables import *
import datetime
import os
import keyboard
import numpy as np
import json


class Training:
    def __init__(self, model_dir: str = None, model_time: str = None):
        self.best_score = -np.inf
        self.scores_history = []
        self.rewards_lst_of_types = []
        self.info = None
        self.main_folder = None
        self.recorder = None
        self.env = None
        self.model_dir = model_dir
        self.model_time = model_time
        self.fig, self.axs = plt.subplots(nrows=3, ncols=1)
        self.legend_ax1 = self.axs[1].legend(loc='lower left')
        self.legend_ax2 = self.axs[2].legend(loc='lower left')

    def train(self):
        self.create_main_folder()
        self.create_log_file()
        self.create_env_and_agent()
        self.print_controls()
        self.print_summaries()
        self.start_recording()
        self.train_model()
        self.recorder.close()

    def plot_learning_curve(self, ax) -> None:
        """
        This functions plots the learning curve of the truck, by showing it's average
        reward over the previous 100 episodes

        @param ax: will plot the graph on that ax
        """
        running_avg = np.zeros(len(self.scores_history))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(self.scores_history[max(0, i-AVE_SCORE_LAST_EP):(i+1)])
        df = pd.DataFrame(running_avg)
        df.plot(ax=ax, title=f'Previous {AVE_SCORE_LAST_EP} average reward', legend=False, figsize=(20, 15), grid=True)

    def plot_reward_types(self, axes) -> None:
        """
        This functions plots the learning curve of the truck by types, by showing what types of
        rewards contribute the most to the total reward of an episode.

        @param axes: will plot the graphs on these axes
        """
        dict_total_reward_per_type = \
            set().union(*(rewards_dict_for_ep.keys() for rewards_dict_for_ep in self.rewards_lst_of_types))
        new_rewards_dict = {}
        for reward_type in dict_total_reward_per_type:
            list_of_rewards_for_a_type = []
            for rewards_dict_for_1ep in self.rewards_lst_of_types:
                if reward_type in rewards_dict_for_1ep:
                    list_of_rewards_for_a_type.append(rewards_dict_for_1ep[reward_type])
                else:
                    list_of_rewards_for_a_type.append(0)
            new_rewards_dict[reward_type] = list_of_rewards_for_a_type
        df = pd.DataFrame(new_rewards_dict)
        df.plot.area(ax=axes[1], title='Reward by types stackplot', figsize=(20, 15), grid=True)
        df.plot(ax=axes[2], title='Reward by types lineplot', figsize=(20, 15), grid=True)

    def create_plot(self, ep) -> None:
        if not ep % AGGREGATE_STATS_PER:
            self.legend_ax1.remove()
            self.legend_ax2.remove()
            self.plot_learning_curve(self.axs[0])
            self.plot_reward_types(self.axs)
            self.fig.savefig(f"{self.main_folder}/Rewards_plots.png")
            print(f'Saving graph')

    def print_summaries(self) -> None:
        """
        This function prints information on the actor and 2 critics.
        """
        if PRINT_SUMMARIES:
            print(self.agent.actor.summary())
            print(self.agent.critic_1.summary())
            print(self.agent.critic_2.summary())

    def create_env_and_agent(self) -> None:
        """
        Creates the Environment and the Agent, and returns them in a tuple

        @param: None
        @return: Tuple[Environment, Agent]
        """
        self.env = Environment()
        if self.model_dir is not None:
            with open(f'{self.model_dir}/log.json', "r") as f:
                model_info = json.load(f)
            print(f"{self.model_dir}/{self.model_time}")
            self.agent = Agent(alpha=model_info['ALPHA'], beta=model_info['BETA'],
                               input_dims=self.env.observation_space.shape, tau=model_info['TAU'],
                               env=self.env, batch_size=model_info['BATCH_SIZE'], warmup=0,
                               n_actions=self.env.action_space.shape[0], noise=model_info['NOISE'],
                               load_models_from_file=f"{self.model_dir}/{self.model_time}")

        else:
            self.agent = Agent(alpha=ALPHA, beta=BETA,
                               input_dims=self.env.observation_space.shape, tau=TAU,
                               env=self.env, batch_size=BATCH_SIZE, warmup=WARMUP,
                               n_actions=self.env.action_space.shape[0], noise=NOISE)

    def create_log_file(self) -> None:
        data = {'TOTAL_EPISODES': TOTAL_EPISODES,
                'TOTAL_STEPS_PER_EPISODE': TOTAL_STEPS_PER_EPISODE,
                'ALPHA': ALPHA, 'BETA': BETA, 'TAU': TAU, 'GAMMA': GAMMA,
                'UPDATE_ACTOR_INTERVAL': UPDATE_ACTOR_INTERVAL,
                'WARMUP': WARMUP, 'MAX_MEM_SIZE': MAX_MEM_SIZE,
                'BATCH_SIZE': BATCH_SIZE, 'NOISE': NOISE,
                'ACTOR_HIDDEN_LAYERS': ACTOR_HIDDEN_LAYERS,
                'CRITIC_1_HIDDEN_LAYERS': CRITIC_1_HIDDEN_LAYERS,
                'CRITIC_2_HIDDEN_LAYERS': CRITIC_2_HIDDEN_LAYERS,
                'TD3_ACTOR_HIDDEN_ACTIVATION': TD3_ACTOR_HIDDEN_ACTIVATION,
                'TD3_CRITIC_HIDDEN_ACTIVATION': TD3_CRITIC_HIDDEN_ACTIVATION,
                'KERNEL_INITIALIZATION_MINVAL': KERNEL_INITIALIZATION_MINVAL,
                'KERNEL_INITIALIZATION_MAXVAL': KERNEL_INITIALIZATION_MAXVAL,
                'MIN_EPSILON': MIN_EPSILON, 'MAX_EPSILON': MAX_EPSILON,
                'Other comments': None}

        with open(f"{self.main_folder}/log.json", "w") as f:
            json.dump(data, f)

    def save_models(self, ep: int, avg_score: np.array) -> np.array:
        """
        Saves the models every 100 episodes, and checks every 5 episodes whether the average is higher than it ever was,
        and if so it saves the models.

        @param ep: the episode number
        @param avg_score: average reward score of the last 100 episodes
        """
        # save model when avg_score is highest ever if the episode number is a multiple of 10
        if avg_score > self.best_score and ep % 10 == 0:
            self.best_score = avg_score
            print('Saving models')
            self.agent.save_models(self.main_folder, ep, avg_score)

        # Saves model per x episode
        if not len(self.scores_history)-1 % SAVE_MODEL_PER:
            print('Saving models')
            self.agent.save_models(self.main_folder, ep, avg_score)

        print(f'Ep{ep}|score:{self.scores_history[-1]:.0f}|avg score:{avg_score:.0f}')

    def create_main_folder(self) -> None:
        self.main_folder = f"td3_models/model_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.recording_folder = f"C:/CM Labs/{self.main_folder}"
        os.mkdir(path=self.main_folder)
        os.mkdir(path=f"{self.main_folder}/recordings")
        os.mkdir(path=f"{self.recording_folder}")

    def start_recording(self) -> None:
        self.env.recorder.open(f"{self.recording_folder}/ep{self.env.ep}.vxrec")
        self.env.recorder.record()

    def stop_recording(self) -> None:
        self.env.recorder.stop()
        self.env.recorder.close()

    def train_model(self) -> None:
        for ep in range(TOTAL_EPISODES):
            self.env.ep = ep
            observation = self.env.reset()
            done = False
            score = 0

            if ep % SAVE_RECORDING_PER == 0:
                self.start_recording()

            while not done:
                action = self.agent.choose_action(observation)
                action_modified_np = tf.Variable(action).numpy()

                if keyboard.is_pressed('esc'):
                    self.env.render(False, False)
                if keyboard.is_pressed('f5'):
                    self.env.render(True, True)
                if keyboard.is_pressed('up'):
                    action_modified_np[0] = 1
                if keyboard.is_pressed('down'):
                    action_modified_np[0] = -1
                if keyboard.is_pressed('right'):
                    action_modified_np[1] = 1
                if keyboard.is_pressed('left'):
                    action_modified_np[1] = -1
                action_modified_np = tf.constant(action_modified_np)

                new_observation, reward, done, self.info = self.env.step(action_modified_np)
                self.agent.remember(observation, action_modified_np, reward, new_observation, done)
                self.agent.learn()
                score += reward
                observation = new_observation

                self.env.HUD_interface.getInputContainer()["Episode"].value = str(ep)
                self.env.HUD_interface.getInputContainer()["Reward"].value = str(round(score, 4))
                self.env.HUD_interface.getInputContainer()["Noise_1"].value = str(round(self.agent.noisy[0], 4))
                self.env.HUD_interface.getInputContainer()["Noise_2"].value = str(round(self.agent.noisy[1], 4))

            self.stop_recording()

            self.scores_history.append(score)
            self.rewards_lst_of_types.append(self.info)

            if ep > AVE_SCORE_AFTER_EP:
                avg_score = np.mean(self.scores_history[-AVE_SCORE_LAST_EP:])
            else:
                avg_score = -np.inf

            self.save_models(ep, avg_score)
            self.create_plot(ep)

    @staticmethod
    def print_controls() -> None:
        """
        Function that prints the different controls that can be used during the training process
        """
        print("*** LEARNING PROCESS STARTING ***")
        print("*** CONTROLS :")
        print("*** ESC: Disable Rendering")
        print("*** F5: Enable Rendering")
        print("*** UP: Forwards")
        print("*** DOWN: Backwards")
        print("*** LEFT: Turn left")
        print("*** RIGHT: Turn right")
        print("*********************************\n")


def main():
    """
    Creates Environment and Agent, creates proper files and folders, and starts learning
    """
    training = Training()
    # "C:/Users/rl-truck-summer-2021/Documents/GitHub/RL-TRUCK/PythonFiles/td3_models/model_2021-08-12_21-18-30",
    # "models/ep90_avg_rew-733")
    training.train()


if __name__ == "__main__":
    main()
