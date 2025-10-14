#!/usr/bin/env python3
"""
Demo script for RL-based split point selection.

Tests the RL module components without full training.
"""

import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from edge.rl_splitting import SplitPointEnv, PPOSplitAgent, SplitController
from edge.resource_monitor import ResourceMonitor
from models.mobilenet_v3 import MobileNetV3Small


def test_environment():
    """Test RL environment."""
    print("=" * 60)
    print("Testing SplitPointEnv")
    print("=" * 60)
    
    # Load config
    with open('configs/streamsplit.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    resource_monitor = ResourceMonitor(config)
    resource_monitor.start()
    
    model = MobileNetV3Small(
        width_mult=config['server']['encoder']['width_mult'],
        embedding_dim=config['server']['encoder']['embedding_dim']
    )
    
    # Create environment
    env = SplitPointEnv(config, resource_monitor, model)
    
    print(f"\n✓ Environment created:")
    print(f"  State dimension: {env.state_dim}")
    print(f"  Number of actions: {env.num_actions}")
    print(f"  Split points: {env.split_points}")
    print(f"  Max episode steps: {env.max_episode_steps}")
    
    # Test reset
    state = env.reset()
    print(f"\n✓ Initial state:")
    print(f"  Shape: {state.shape}")
    print(f"  Range: [{state.min():.3f}, {state.max():.3f}]")
    print(f"  Values: {state}")
    
    # Test episode
    print(f"\n✓ Running test episode:")
    episode_reward = 0
    
    for step in range(10):
        # Random action
        action = np.random.randint(0, env.num_actions)
        next_state, reward, done, info = env.step(action)
        
        episode_reward += reward
        
        print(f"  Step {step}: action={action}, "
              f"split_layer={info['split_layer']}, "
              f"reward={reward:.3f}, "
              f"accuracy={info['accuracy']:.3f}, "
              f"latency={info['latency']:.4f}")
        
        state = next_state
        
        if done:
            break
    
    print(f"\n  Total reward: {episode_reward:.3f}")
    
    resource_monitor.stop()
    print("\n✓ Environment test passed!")
    return True


def test_agent():
    """Test PPO agent."""
    print("\n" + "=" * 60)
    print("Testing PPOSplitAgent")
    print("=" * 60)
    
    # Load config
    with open('configs/streamsplit.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    resource_monitor = ResourceMonitor(config)
    resource_monitor.start()
    
    model = MobileNetV3Small(
        width_mult=config['server']['encoder']['width_mult'],
        embedding_dim=config['server']['encoder']['embedding_dim']
    )
    
    # Create environment and agent
    env = SplitPointEnv(config, resource_monitor, model)
    agent = PPOSplitAgent(config, env)
    
    print(f"\n✓ Agent created:")
    print(f"  Policy parameters: "
          f"{sum(p.numel() for p in agent.policy.parameters())}")
    print(f"  Learning rate: {agent.config.get('learning_rate', 3e-4)}")
    print(f"  Gamma: {agent.gamma}")
    print(f"  Epsilon clip: {agent.epsilon_clip}")
    
    # Test action selection
    print(f"\n✓ Testing action selection:")
    state = env.reset()
    
    for i in range(5):
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(reward, done)
        
        print(f"  Selection {i}: action={action}, "
              f"split_layer={info['split_layer']}, reward={reward:.3f}")
        
        state = next_state
    
    # Test update
    print(f"\n✓ Testing policy update:")
    print(f"  Buffer size: {len(agent.states)} transitions")
    
    metrics = agent.update()
    
    if metrics:
        print(f"  Policy loss: {metrics['policy_loss']:.4f}")
        print(f"  Value loss: {metrics['value_loss']:.4f}")
        print(f"  Entropy: {metrics['entropy']:.4f}")
    
    resource_monitor.stop()
    print("\n✓ Agent test passed!")
    return True


def test_controller():
    """Test split controller."""
    print("\n" + "=" * 60)
    print("Testing SplitController")
    print("=" * 60)
    
    # Load config
    with open('configs/streamsplit.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    resource_monitor = ResourceMonitor(config)
    resource_monitor.start()
    
    # Create controller (without trained policy)
    controller = SplitController(config, resource_monitor, policy_path=None)
    
    print(f"\n✓ Controller created:")
    print(f"  Split points: {controller.split_points}")
    print(f"  Policy parameters: "
          f"{sum(p.numel() for p in controller.policy.parameters())}")
    
    # Test split decisions
    print(f"\n✓ Testing split decisions:")
    
    for i in range(10):
        # Simulate different network conditions
        network_metrics = {
            'latency': np.random.uniform(0.1, 1.0),
            'bandwidth': np.random.uniform(0.3, 1.0)
        }
        
        split_layer = controller.get_split_layer(network_metrics)
        
        if i < 5:  # Show first 5
            print(f"  Decision {i}: split_layer={split_layer}, "
                  f"latency={network_metrics['latency']:.3f}, "
                  f"bandwidth={network_metrics['bandwidth']:.3f}")
    
    # Get statistics
    stats = controller.get_statistics()
    print(f"\n✓ Controller statistics:")
    print(f"  Total decisions: {stats['num_decisions']}")
    print(f"  Avg split layer: {stats['avg_split_layer']:.2f}")
    print(f"  Std split layer: {stats['std_split_layer']:.2f}")
    print(f"  Action distribution: {stats['action_distribution']}")
    print(f"  Avg resource pressure: {stats['avg_resource_pressure']:.3f}")
    
    resource_monitor.stop()
    print("\n✓ Controller test passed!")
    return True


def test_actor_critic_network():
    """Test Actor-Critic network architecture."""
    print("\n" + "=" * 60)
    print("Testing ActorCritic Network")
    print("=" * 60)
    
    from edge.rl_splitting import ActorCritic
    
    # Create network
    state_dim = 6
    action_dim = 6
    hidden_dim = 128
    
    network = ActorCritic(state_dim, action_dim, hidden_dim)
    
    print(f"\n✓ Network architecture:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in network.parameters())}")
    
    # Test forward pass
    batch_size = 32
    states = torch.randn(batch_size, state_dim)
    
    action_probs, state_values = network(states)
    
    print(f"\n✓ Forward pass:")
    print(f"  Input shape: {states.shape}")
    print(f"  Action probs shape: {action_probs.shape}")
    print(f"  State values shape: {state_values.shape}")
    print(f"  Action probs sum: {action_probs.sum(dim=1).mean():.3f} "
          "(should be ~1.0)")
    
    # Test action sampling
    single_state = torch.randn(state_dim)
    action, log_prob, value = network.act(single_state)
    
    print(f"\n✓ Action sampling:")
    print(f"  Sampled action: {action}")
    print(f"  Log probability: {log_prob.item():.3f}")
    print(f"  State value: {value.item():.3f}")
    
    # Test evaluation
    actions = torch.randint(0, action_dim, (batch_size,))
    log_probs, values, entropy = network.evaluate(states, actions)
    
    print(f"\n✓ Evaluation:")
    print(f"  Log probs shape: {log_probs.shape}")
    print(f"  Values shape: {values.shape}")
    print(f"  Entropy shape: {entropy.shape}")
    print(f"  Mean entropy: {entropy.mean():.3f}")
    
    print("\n✓ Network test passed!")
    return True


def main():
    """Run all RL module tests."""
    print("\n" + "=" * 60)
    print("StreamSplit RL Module Tests")
    print("=" * 60)
    
    try:
        # Test individual components
        network_ok = test_actor_critic_network()
        env_ok = test_environment()
        agent_ok = test_agent()
        controller_ok = test_controller()
        
        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Actor-Critic Network: {'✓ PASS' if network_ok else '✗ FAIL'}")
        print(f"Environment: {'✓ PASS' if env_ok else '✗ FAIL'}")
        print(f"PPO Agent: {'✓ PASS' if agent_ok else '✗ FAIL'}")
        print(f"Split Controller: {'✓ PASS' if controller_ok else '✗ FAIL'}")
        
        if all([network_ok, env_ok, agent_ok, controller_ok]):
            print("\n✓ All RL module tests passed!")
            return 0
        else:
            print("\n✗ Some tests failed!")
            return 1
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
