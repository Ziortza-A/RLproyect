import numpy as np
import matplotlib.pyplot as plt
import optuna
from env import GridworldEnv
import tools

def get_action(greedy_action, epsilon, n_actions):
    prob = np.random.rand()
    if prob < epsilon:
        return np.random.choice(n_actions)
    else:
        return greedy_action


class TabularRLAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, min_epsilon=0.01, decay_rate=0.995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.policy = np.zeros(self.n_states, dtype=int)


    def update_policy(self):
        for s in range(self.n_states):
            self.policy[s] = np.argmax(self.Q[s])



    def train_q_learning_episode(self, env, max_steps=200):
        s,_ = env.reset()
        done=False
        step_i=0
        total_reward=0.0

        while not done and step_i < max_steps:
            # Action selection following epsilon-greedy policy
            a = get_action(np.argmax(self.Q[s]), self.epsilon, self.n_actions)

            # Step in environment
            s_prime, rwd, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total_reward += rwd

            # Q-Learning (Off-policy) update rule
            future_q = 0.0 if terminated else np.max(self.Q[s_prime])
            td_target = rwd + self.gamma * future_q
            td_error = td_target - self.Q[s][a]
            self.Q[s][a] += self.alpha * td_error

            s = s_prime
            step_i += 1


        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

        return total_reward

    def train(self, env, num_episodes=500, max_steps=200):
        rewards_history = []
        for ep_i in range(num_episodes):
            reward = self.train_q_learning_episode(env, max_steps=max_steps)
            rewards_history.append(reward)
        self.update_policy()
        return rewards_history



# OPTUNA HYPERPARAMETER OPTIMIZATION

def objective(trial, is_slippery=False):
    alpha = trial.suggest_float("alpha", 0.01, 0.5, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    decay_rate = trial.suggest_float("decay_rate", 0.95, 0.999)


    env = GridworldEnv(is_slippery=is_slippery)
    agent = TabularRLAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=1.0,
        decay_rate=decay_rate
    )

    num_eval_runs = 20
    episodes_per_run = 300
    mean_rewards = []

    for _ in range(num_eval_runs):
        agent.Q.fill(0)
        agent.epsilon = 1.0
        rewards = agent.train(env, num_episodes=episodes_per_run)
        mean_rewards.append(np.mean(rewards[-50:]))

    return np.mean(mean_rewards)



def run_hyperparameter_tuning(is_slippery=False, n_trials=50):
    print(f"\n--- Optuna (is_slippery={is_slippery}) ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, is_slippery=is_slippery), n_trials=n_trials)

    print("hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.4f}")
    print(f"Best mean reward: {study.best_value:.4f}")

    return study.best_params


if __name__== "__main__":
    # Run Optuna
    best_params = run_hyperparameter_tuning(is_slippery=False, n_trials=20)

    #train model with best hyperparameters obrained from optuna
    env = GridworldEnv(is_slippery=False)
    agent = TabularRLAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        **best_params
    )

    rewards = agent.train(env, num_episodes=500)

    #Value function and visualization
    V = np.max(agent.Q, axis=1)
    tools.plot_value_function(V, env, title="Q-Learning Value Function V(s)")
    tools.plot_policy(agent.policy, env, title="Q-Learning Optimal Policy π(s)")