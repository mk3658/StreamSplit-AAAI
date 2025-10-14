#!/usr/bin/env python3
"""
Training script for RL-based split point agent.

Trains a PPO agent to learn optimal computation splitting decisions
based on resource constraints and performance metrics.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from edge.rl_splitting import SplitPointEnv, PPOSplitAgent, SplitController
from edge.resource_monitor import ResourceMonitor
from models.mobilenet_v3 import MobileNetV3Small
from utils.logger import Logger
from utils.device import get_device, print_device_info, optimize_for_device


def plot_training_curves(rewards, policy_losses, value_losses, save_path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Rewards
    axes[0].plot(rewards)
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True, alpha=0.3)
    
    # Policy loss
    axes[1].plot(policy_losses)
    axes[1].set_title('Policy Loss')
    axes[1].set_xlabel('Update')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    
    # Value loss
    axes[2].plot(value_losses)
    axes[2].set_title('Value Loss')
    axes[2].set_xlabel('Update')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {save_path}")


def evaluate_policy(agent, env, num_episodes=10):
    """
    Evaluate trained policy.
    
    Args:
        agent: PPOSplitAgent
        env: SplitPointEnv
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary of evaluation metrics
    """
    episode_rewards = []
    split_decisions = []
    accuracies = []
    latencies = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_splits = []
        
        for _ in range(env.max_episode_steps):
            # Select action (greedy, no exploration)
            state_tensor = torch.FloatTensor(state).to(agent.device)
            with torch.no_grad():
                action_probs, _ = agent.policy(state_tensor.unsqueeze(0))
                action = torch.argmax(action_probs, dim=-1).item()
            
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_splits.append(info['split_layer'])
            
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        split_decisions.extend(episode_splits)
        
        # Record final performance
        accuracies.append(env.current_accuracy)
        latencies.append(env.current_latency)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_accuracy': np.mean(accuracies),
        'mean_latency': np.mean(latencies),
        'split_distribution': np.bincount(split_decisions, 
                                         minlength=env.num_actions).tolist()
    }


def train_rl_agent(config, args):
    """
    Train RL agent for split point selection (GPU-enabled).
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    # Auto-detect and set best available device
    device = get_device(force_cpu=args.force_cpu)
    print_device_info(device)
    optimize_for_device(device)
    
    # Update config with detected device
    config['experiment']['device'] = str(device)
    
    # Create directories
    log_dir = Path(config['experiment']['log_dir']) / 'rl_training'
    checkpoint_dir = Path(config['experiment']['checkpoint_dir']) / 'rl'
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = Logger(log_dir, 'rl_training')
    
    # Initialize resource monitor
    resource_monitor = ResourceMonitor(config)
    resource_monitor.start()
    
    # Initialize model (for split point info) and move to device
    print("Initializing model...")
    model = MobileNetV3Small(
        width_mult=config['server']['encoder']['width_mult'],
        embedding_dim=config['server']['encoder']['embedding_dim']
    ).to(device)
    print(f"✓ Model moved to {device}")
    
    # Create environment
    env = SplitPointEnv(config, resource_monitor, model)
    
    # Create agent (will use device from config)
    agent = PPOSplitAgent(config, env)
    
    print("=" * 60)
    print("RL-based Split Point Training")
    print("=" * 60)
    print(f"State dimension: {env.state_dim}")
    print(f"Number of actions: {env.num_actions}")
    print(f"Split points: {env.split_points}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max episode steps: {env.max_episode_steps}")
    print(f"Update frequency: every {args.update_frequency} steps")
    print("=" * 60)
    
    # Training metrics
    episode_rewards = []
    policy_losses = []
    value_losses = []
    entropies = []
    
    total_steps = 0
    best_reward = -float('inf')
    
    # Training loop
    pbar = tqdm(range(args.num_episodes), desc="Training")
    for episode in pbar:
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Episode rollout
        for step in range(env.max_episode_steps):
            # Select action
            action = agent.select_action(state)
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(reward, done)
            
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            state = next_state
            
            # Update policy
            if total_steps % args.update_frequency == 0:
                metrics = agent.update()
                if metrics:
                    policy_losses.append(metrics['policy_loss'])
                    value_losses.append(metrics['value_loss'])
                    entropies.append(metrics['entropy'])
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Update progress bar
        pbar.set_postfix({
            'reward': f'{episode_reward:.3f}',
            'length': episode_length,
            'best': f'{best_reward:.3f}'
        })
        
        # Logging
        if (episode + 1) % args.log_frequency == 0:
            avg_reward = np.mean(episode_rewards[-args.log_frequency:])
            
            logger.log({
                'episode': episode + 1,
                'avg_reward': avg_reward,
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'total_steps': total_steps,
                'policy_loss': policy_losses[-1] if policy_losses else 0,
                'value_loss': value_losses[-1] if value_losses else 0,
                'entropy': entropies[-1] if entropies else 0
            })
            
            print(f"\nEpisode {episode + 1}/{args.num_episodes}")
            print(f"  Avg Reward (last {args.log_frequency}): {avg_reward:.3f}")
            print(f"  Episode Reward: {episode_reward:.3f}")
            print(f"  Total Steps: {total_steps}")
        
        # Evaluation
        if (episode + 1) % args.eval_frequency == 0:
            print(f"\nEvaluating policy at episode {episode + 1}...")
            eval_metrics = evaluate_policy(agent, env, num_episodes=10)
            
            print(f"  Mean Reward: {eval_metrics['mean_reward']:.3f} "
                  f"± {eval_metrics['std_reward']:.3f}")
            print(f"  Mean Accuracy: {eval_metrics['mean_accuracy']:.3f}")
            print(f"  Mean Latency: {eval_metrics['mean_latency']:.4f}")
            print(f"  Split Distribution: {eval_metrics['split_distribution']}")
            
            logger.log({
                'episode': episode + 1,
                'eval_mean_reward': eval_metrics['mean_reward'],
                'eval_std_reward': eval_metrics['std_reward'],
                'eval_mean_accuracy': eval_metrics['mean_accuracy'],
                'eval_mean_latency': eval_metrics['mean_latency']
            })
            
            # Save best model
            if eval_metrics['mean_reward'] > best_reward:
                best_reward = eval_metrics['mean_reward']
                best_path = checkpoint_dir / 'best_policy.pt'
                agent.save(str(best_path))
                print(f"  ✓ Saved best policy to {best_path}")
        
        # Checkpoint saving
        if (episode + 1) % args.checkpoint_frequency == 0:
            checkpoint_path = checkpoint_dir / f'policy_ep{episode+1}.pt'
            agent.save(str(checkpoint_path))
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    eval_metrics = evaluate_policy(agent, env, num_episodes=50)
    
    print(f"Mean Reward: {eval_metrics['mean_reward']:.3f} "
          f"± {eval_metrics['std_reward']:.3f}")
    print(f"Mean Accuracy: {eval_metrics['mean_accuracy']:.3f}")
    print(f"Mean Latency: {eval_metrics['mean_latency']:.4f}")
    print(f"Split Layer Distribution:")
    for i, count in enumerate(eval_metrics['split_distribution']):
        split_layer = env.split_points[i]
        percentage = count / sum(eval_metrics['split_distribution']) * 100
        print(f"  Layer {split_layer}: {count} ({percentage:.1f}%)")
    
    # Save final model
    final_path = checkpoint_dir / 'final_policy.pt'
    agent.save(str(final_path))
    print(f"\nSaved final policy to {final_path}")
    
    # Plot training curves
    plot_path = log_dir / 'training_curves.png'
    plot_training_curves(episode_rewards, policy_losses, 
                        value_losses, str(plot_path))
    
    # Cleanup
    resource_monitor.stop()
    logger.close()
    
    print("\n✓ Training completed successfully!")


def test_split_controller(config, policy_path):
    """
    Test trained split controller.
    
    Args:
        config: Configuration dictionary
        policy_path: Path to trained policy
    """
    print("=" * 60)
    print("Testing Split Controller")
    print("=" * 60)
    
    # Initialize resource monitor
    resource_monitor = ResourceMonitor(config)
    resource_monitor.start()
    
    # Create controller
    controller = SplitController(config, resource_monitor, policy_path)
    
    # Test decisions under different conditions
    print("\nSimulating split decisions...")
    
    for i in range(20):
        # Simulate varying network conditions
        network_metrics = {
            'latency': np.random.uniform(0.1, 1.0),
            'bandwidth': np.random.uniform(0.3, 1.0)
        }
        
        split_layer = controller.get_split_layer(network_metrics)
        stats = controller.get_statistics()
        
        if (i + 1) % 5 == 0:
            print(f"\nDecision {i+1}:")
            print(f"  Split Layer: {split_layer}")
            print(f"  Network Latency: {network_metrics['latency']:.3f}")
            print(f"  Network Bandwidth: {network_metrics['bandwidth']:.3f}")
            print(f"  Avg Resource Pressure: "
                  f"{stats['avg_resource_pressure']:.3f}")
    
    # Final statistics
    final_stats = controller.get_statistics()
    print("\n" + "=" * 60)
    print("Split Controller Statistics")
    print("=" * 60)
    print(f"Total Decisions: {final_stats['num_decisions']}")
    print(f"Avg Split Layer: {final_stats['avg_split_layer']:.2f}")
    print(f"Std Split Layer: {final_stats['std_split_layer']:.2f}")
    print(f"Action Distribution: {final_stats['action_distribution']}")
    print(f"Avg Resource Pressure: {final_stats['avg_resource_pressure']:.3f}")
    
    resource_monitor.stop()
    print("\n✓ Controller test completed!")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train RL agent for split point selection (GPU-enabled)'
    )
    parser.add_argument('--config', type=str,
                       default='configs/streamsplit.yaml',
                       help='Path to configuration file')
    parser.add_argument('--num_episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--update_frequency', type=int, default=256,
                       help='Policy update frequency (steps)')
    parser.add_argument('--log_frequency', type=int, default=10,
                       help='Logging frequency (episodes)')
    parser.add_argument('--eval_frequency', type=int, default=50,
                       help='Evaluation frequency (episodes)')
    parser.add_argument('--checkpoint_frequency', type=int, default=100,
                       help='Checkpoint saving frequency (episodes)')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    parser.add_argument('--test', action='store_true',
                       help='Test trained controller')
    parser.add_argument('--policy_path', type=str,
                       default='checkpoints/rl/best_policy.pt',
                       help='Path to trained policy for testing')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds
    seed = config['experiment']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    if args.test:
        # Test mode
        test_split_controller(config, args.policy_path)
    else:
        # Training mode
        train_rl_agent(config, args)


if __name__ == '__main__':
    main()
