"""
Logging utilities for StreamSplit.
"""

import os
import json
from datetime import datetime
from typing import Dict


class Logger:
    """Simple logger for experiments."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """Initialize logger."""
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            log_dir, f"{experiment_name}_{timestamp}.log"
        )
        
        # Initialize
        self.metrics_history = []
        
    def log(self, message: str):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        print(formatted_message)
        
        with open(self.log_file, 'a') as f:
            f.write(formatted_message + '\n')
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for an epoch."""
        self.metrics_history.append({
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        # Log to file
        metrics_str = ', '.join([f"{k}: {v:.4f}" 
                                for k, v in metrics.items()])
        self.log(f"Epoch {epoch} - {metrics_str}")
    
    def save_metrics(self):
        """Save metrics history to JSON."""
        metrics_file = os.path.join(
            self.log_dir, f"{self.experiment_name}_metrics.json"
        )
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.log(f"Saved metrics to {metrics_file}")
