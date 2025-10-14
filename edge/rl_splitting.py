"""
RL-based Computation Splitting Module

Implements PPO-based adaptive split point selection for edge-server
computation partitioning based on resource constraints and performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class SplitPointEnv:
    """
    Environment for learning optimal split points in edge-server architecture.
    
    State: Resource metrics (CPU, memory, battery, network)
    Action: Split layer index (0=edge only, max=server only)
    Reward: Weighted combination of accuracy, latency, and resource usage
    """
    
    def __init__(self, config: Dict, resource_monitor, model):
        """
        Initialize split point environment.
        
        Args:
            config: Configuration dictionary
            resource_monitor: ResourceMonitor instance for state observation
            model: Neural network model with split points
        """
        self.config = config['splitting']
        self.resource_monitor = resource_monitor
        self.model = model
        
        # Split points (MobileNetV3-Small has stages at indices:
        # 0, 2, 4, 7, 11, 13)
        split_points_list = config['server']['encoder'].get(
            'split_points', [0, 2, 4, 7, 11, 13]
        )
        self.split_points = split_points_list
        self.num_actions = len(self.split_points)
        
        # State space: CPU, memory, battery, thermal, network latency, network bandwidth
        self.state_dim = 6
        
        # Reward weights
        self.alpha_accuracy = self.config.get('alpha_accuracy', 0.5)
        self.alpha_latency = self.config.get('alpha_latency', 0.3)
        self.alpha_resource = self.config.get('alpha_resource', 0.2)
        
        # Performance tracking
        self.current_accuracy = 0.0
        self.current_latency = 0.0
        self.current_resource_usage = 0.0
        
        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = self.config.get('max_episode_steps', 100)
        
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial state observation
        """
        self.episode_step = 0
        self.current_accuracy = 0.0
        self.current_latency = 0.0
        self.current_resource_usage = 0.0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state observation.
        
        Returns:
            State vector with normalized resource metrics
        """
        # Get resource metrics
        metrics = self.resource_monitor.get_state()
        
        # Normalize to [0, 1]
        state = np.array([
            metrics['cpu_util'] / 100.0,           # CPU usage
            metrics['mem_usage'] / 100.0,          # Memory usage
            metrics['battery_level'] / 100.0,      # Battery level
            1.0 if metrics['thermal_throttling'] else 0.0,  # Thermal state
            np.random.uniform(0.1, 1.0),           # Simulated network latency
            np.random.uniform(0.5, 1.0),           # Simulated network bandwidth
        ], dtype=np.float32)
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state, reward, done flag.
        
        Args:
            action: Split point index
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.episode_step += 1
        
        # Get split layer
        split_layer = self.split_points[action]
        
        # Simulate performance metrics based on split point
        # Earlier splits = more edge computation = higher resource usage
        # Later splits = more server computation = higher latency
        edge_ratio = action / (self.num_actions - 1)  # 0 = all edge, 1 = all server
        
        # Get current resource state
        state = self._get_state()
        resource_pressure = (state[0] + state[1]) / 2.0  # CPU + memory
        
        # Compute metrics
        # Accuracy: slightly better with more computation (server)
        self.current_accuracy = 0.85 + 0.10 * (1 - edge_ratio)
        
        # Latency: increases with server computation (network overhead)
        network_latency = state[4]
        self.current_latency = 0.01 * edge_ratio + 0.1 * (1 - edge_ratio) * network_latency
        
        # Resource usage: higher with edge computation
        self.current_resource_usage = resource_pressure * (1 - edge_ratio) * 0.3
        
        # Compute reward
        reward = self._compute_reward(action, resource_pressure)
        
        # Check termination
        done = self.episode_step >= self.max_episode_steps
        
        # Next state
        next_state = self._get_state()
        
        info = {
            'split_layer': split_layer,
            'accuracy': self.current_accuracy,
            'latency': self.current_latency,
            'resource_usage': self.current_resource_usage,
            'edge_ratio': edge_ratio
        }
        
        return next_state, reward, done, info
    
    def _compute_reward(self, action: int, resource_pressure: float) -> float:
        """
        Compute reward for split decision.
        
        Args:
            action: Split point action
            resource_pressure: Current resource utilization
            
        Returns:
            Scalar reward value
        """
        # Reward components
        r_accuracy = self.current_accuracy  # Higher is better
        r_latency = 1.0 - self.current_latency  # Lower latency = higher reward
        r_resource = 1.0 - self.current_resource_usage  # Lower usage = higher reward
        
        # Penalty for inappropriate split under resource pressure
        # If high resource pressure, should use more server (higher action)
        edge_ratio = action / (self.num_actions - 1)
        if resource_pressure > 0.7 and edge_ratio < 0.3:
            # Using too much edge when resources are scarce
            penalty = -0.5 * (0.7 - edge_ratio)
        else:
            penalty = 0.0
        
        # Weighted combination
        reward = (self.alpha_accuracy * r_accuracy +
                 self.alpha_latency * r_latency +
                 self.alpha_resource * r_resource +
                 penalty)
        
        return reward


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize actor-critic networks.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor-critic.
        
        Args:
            state: State tensor [batch, state_dim]
            
        Returns:
            Tuple of (action_probs, state_value)
        """
        features = self.shared(state)
        action_probs = self.policy(features)
        state_value = self.value(features)
        
        return action_probs, state_value
    
    def act(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: State tensor [state_dim]
            
        Returns:
            Tuple of (action, log_prob, state_value)
        """
        action_probs, state_value = self.forward(state.unsqueeze(0))
        action_probs = action_probs.squeeze(0)
        
        # Sample action
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, state_value.squeeze(0)
    
    def evaluate(self, states: torch.Tensor, 
                actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under current policy.
        
        Args:
            states: State tensor [batch, state_dim]
            actions: Action tensor [batch]
            
        Returns:
            Tuple of (log_probs, state_values, entropy)
        """
        action_probs, state_values = self.forward(states)
        
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, state_values.squeeze(-1), entropy


class PPOSplitAgent:
    """PPO agent for learning optimal split points."""
    
    def __init__(self, config: Dict, env: SplitPointEnv):
        """
        Initialize PPO agent.
        
        Args:
            config: Configuration dictionary
            env: SplitPointEnv instance
        """
        self.config = config['splitting']
        self.env = env
        self.device = torch.device(config['experiment']['device'])
        
        # Networks
        self.policy = ActorCritic(
            state_dim=env.state_dim,
            action_dim=env.num_actions,
            hidden_dim=self.config.get('hidden_dim', 128)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.get('learning_rate', 3e-4)
        )
        
        # PPO hyperparameters
        self.gamma = self.config.get('gamma', 0.99)
        self.lambda_gae = self.config.get('lambda_gae', 0.95)
        self.epsilon_clip = self.config.get('epsilon_clip', 0.2)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        self.value_coef = self.config.get('value_coef', 0.5)
        self.max_grad_norm = self.config.get('max_grad_norm', 0.5)
        
        # Training parameters
        self.update_epochs = self.config.get('update_epochs', 10)
        self.batch_size = self.config.get('batch_size', 64)
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action index
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.act(state_tensor)
        
        # Store experience
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())
        
        return action
    
    def store_transition(self, reward: float, done: bool):
        """
        Store transition in buffer.
        
        Args:
            reward: Reward received
            done: Episode termination flag
        """
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_gae(self, rewards: List[float], 
                   values: List[float], 
                   dones: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards
            values: List of state values
            dones: List of done flags
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        
        advantage = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            
            # GAE
            advantage = delta + self.gamma * self.lambda_gae * advantage * mask
            
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])
            
            next_value = values[t]
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self):
        """Update policy using PPO."""
        if len(self.states) == 0:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            self.rewards, self.values, self.dones
        )
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for _ in range(self.update_epochs):
            # Mini-batch updates
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, state_values, entropy = self.policy.evaluate(
                    batch_states, batch_actions
                )
                
                # Ratio for PPO
                ratios = torch.exp(log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 
                                   1 + self.epsilon_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(state_values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.value_coef * value_loss + 
                       self.entropy_coef * entropy_loss)
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        
        return {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1)
        }
    
    def save(self, path: str):
        """Save policy checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load policy checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class SplitController:
    """Controller for runtime split point decisions using trained RL agent."""
    
    def __init__(self, config: Dict, resource_monitor, policy_path: Optional[str] = None):
        """
        Initialize split controller.
        
        Args:
            config: Configuration dictionary
            resource_monitor: ResourceMonitor instance
            policy_path: Path to trained policy checkpoint
        """
        self.config = config
        self.resource_monitor = resource_monitor
        self.device = torch.device(config['experiment']['device'])
        
        # Split points
        split_points_list = config['server']['encoder'].get(
            'split_points', [0, 2, 4, 7, 11, 13]
        )
        self.split_points = split_points_list
        
        # Policy network
        self.policy = ActorCritic(
            state_dim=6,  # CPU, memory, battery, thermal, latency, bandwidth
            action_dim=len(self.split_points),
            hidden_dim=config['splitting'].get('hidden_dim', 128)
        ).to(self.device)
        
        # Load trained policy
        if policy_path:
            self.load_policy(policy_path)
        
        self.policy.eval()
        
        # Decision history
        self.history = deque(maxlen=100)
        
    def get_split_layer(self, network_metrics: Optional[Dict] = None) -> int:
        """
        Get optimal split layer based on current state.
        
        Args:
            network_metrics: Optional network condition metrics
            
        Returns:
            Split layer index
        """
        # Get resource state
        metrics = self.resource_monitor.get_state()
        
        # Build state vector
        state = np.array([
            metrics['cpu_util'] / 100.0,
            metrics['mem_usage'] / 100.0,
            metrics['battery_level'] / 100.0,
            1.0 if metrics['thermal_throttling'] else 0.0,
            network_metrics.get('latency', 0.5) if network_metrics else 0.5,
            network_metrics.get('bandwidth', 0.8) if network_metrics else 0.8,
        ], dtype=np.float32)
        
        # Get action from policy
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.policy(state_tensor.unsqueeze(0))
            action = torch.argmax(action_probs, dim=-1).item()
        
        split_layer = self.split_points[action]
        
        # Store decision
        self.history.append({
            'state': state,
            'action': action,
            'split_layer': split_layer,
            'resource_pressure': (
                (metrics['cpu_util'] + metrics['mem_usage']) / 200.0
            )
        })
        
        return split_layer
    
    def load_policy(self, path: str):
        """Load trained policy."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"Loaded trained policy from {path}")
    
    def get_statistics(self) -> Dict:
        """Get decision statistics."""
        if not self.history:
            return {}
        
        actions = [h['action'] for h in self.history]
        split_layers = [h['split_layer'] for h in self.history]
        resource_pressures = [h['resource_pressure'] for h in self.history]
        
        return {
            'avg_split_layer': np.mean(split_layers),
            'std_split_layer': np.std(split_layers),
            'action_distribution': np.bincount(actions, minlength=len(self.split_points)).tolist(),
            'avg_resource_pressure': np.mean(resource_pressures),
            'num_decisions': len(self.history)
        }
