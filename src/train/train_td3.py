"""Minimal TD3 training loop.

This module wires together prepared datasets, the portfolio environment, replay
memory, and the TD3 agent. It returns in-memory validation and test evaluation
results, but intentionally does not save artifacts or produce empirical output
files.
"""

from src.backtest.evaluate_agent import evaluate_agent
from src.backtest.compare_policies import compare_agent_to_basic_benchmarks
from src.data.prepare_dataset import prepare_train_validation_test_datasets
from src.env.portfolio_env import PortfolioEnv
from src.memory.replay_buffer import ReplayBuffer
from src.models.td3_agent import TD3Agent
from src.utils.config import load_config
from src.utils.seed import set_seed


def train_td3(config_path: str) -> dict:
    """Run a minimal TD3 training loop and return in-memory results."""
    config = load_config(config_path)
    datasets = prepare_train_validation_test_datasets(config_path)

    return train_td3_on_datasets(datasets, config)


def train_td3_on_datasets(
    datasets: dict,
    config: dict,
) -> dict:
    """Run TD3 training/evaluation using already-prepared datasets."""
    training_config = config["training"]
    environment_config = config["environment"]
    td3_config = config["td3"]

    set_seed(training_config["seed"])

    env = PortfolioEnv(
        returns=datasets["train_returns"],
        features=datasets["train_features"],
        initial_cash=environment_config["initial_cash"],
        transaction_cost=environment_config["transaction_cost"],
        reward_config=config["reward"],
    )
    state_dim = env.observation_dim
    action_dim = env.n_assets
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_learning_rate=td3_config["actor_learning_rate"],
        critic_learning_rate=td3_config["critic_learning_rate"],
        gamma=td3_config["gamma"],
        tau=td3_config["tau"],
        policy_noise=td3_config["policy_noise"],
        noise_clip=td3_config["noise_clip"],
        policy_delay=td3_config["policy_delay"],
    )
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=td3_config["replay_buffer_size"],
    )

    episode_logs = []
    for episode in range(1, training_config["episodes"] + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        latest_losses = {
            "critic_1_loss": None,
            "critic_2_loss": None,
            "actor_loss": None,
        }
        info = {"portfolio_value": env.portfolio_value}
        episode_turnover = []
        episode_transaction_costs = []
        episode_weights = []

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            episode_turnover.append(info["turnover"])
            episode_transaction_costs.append(info["transaction_cost"])
            episode_weights.append(info["weights"])

            if len(replay_buffer) >= td3_config["batch_size"]:
                batch = replay_buffer.sample(td3_config["batch_size"])
                latest_losses = agent.train_step(batch)

            state = next_state
            total_reward += reward
            steps += 1

        final_weights = {
            asset_name: float(weight)
            for asset_name, weight in zip(env.asset_names, episode_weights[-1])
        }
        episode_logs.append(
            {
                "episode": episode,
                "final_portfolio_value": info["portfolio_value"],
                "total_reward": total_reward,
                "steps": steps,
                "average_turnover": sum(episode_turnover) / steps,
                "average_transaction_cost": sum(episode_transaction_costs) / steps,
                "final_weights": final_weights,
                "max_weight": max(final_weights.values()),
                "cash_weight": final_weights.get("CASH", 0.0),
                "critic_1_loss": latest_losses["critic_1_loss"],
                "critic_2_loss": latest_losses["critic_2_loss"],
                "actor_loss": latest_losses["actor_loss"],
            }
        )

    evaluation_kwargs = {
        "initial_cash": environment_config["initial_cash"],
        "transaction_cost": environment_config["transaction_cost"],
        "reward_config": config["reward"],
        "periods_per_year": 52,
        "risk_free_rate": 0.0,
    }
    validation_evaluation = evaluate_agent(
        agent,
        datasets["validation_returns"],
        datasets["validation_features"],
        **evaluation_kwargs,
    )
    test_evaluation = evaluate_agent(
        agent,
        datasets["test_returns"],
        datasets["test_features"],
        **evaluation_kwargs,
    )
    validation_comparison = compare_agent_to_basic_benchmarks(
        agent,
        datasets["validation_returns"],
        datasets["validation_features"],
        **evaluation_kwargs,
    )
    test_comparison = compare_agent_to_basic_benchmarks(
        agent,
        datasets["test_returns"],
        datasets["test_features"],
        **evaluation_kwargs,
    )

    return {
        "agent": agent,
        "replay_buffer": replay_buffer,
        "episode_logs": episode_logs,
        "train_returns": datasets["train_returns"],
        "train_features": datasets["train_features"],
        "validation_returns": datasets["validation_returns"],
        "validation_features": datasets["validation_features"],
        "test_returns": datasets["test_returns"],
        "test_features": datasets["test_features"],
        "validation_evaluation": validation_evaluation,
        "test_evaluation": test_evaluation,
        "validation_comparison": validation_comparison,
        "test_comparison": test_comparison,
    }
