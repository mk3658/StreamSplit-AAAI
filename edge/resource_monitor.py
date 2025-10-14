"""
Resource monitoring for edge devices.
Tracks CPU, memory, battery, and thermal state.
"""

import psutil
import time
from typing import Dict
from threading import Thread, Lock


class ResourceMonitor:
    """Monitor system resources on edge device."""
    
    def __init__(self, config: Dict):
        """
        Initialize resource monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config['edge']['resource_monitor']
        self.poll_interval = self.config['poll_interval_ms'] / 1000.0
        
        # Resource state
        self.state = {
            'cpu_util': 0.0,
            'mem_usage': 0.0,
            'battery_level': 1.0,
            'thermal_throttling': 0.0,
            'timestamp': time.time()
        }
        
        # Thread safety
        self.lock = Lock()
        self.running = False
        self.monitor_thread = None
        
    def start(self):
        """Start monitoring in background thread."""
        if not self.running:
            self.running = True
            self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            # Get current resource usage
            cpu_util = psutil.cpu_percent(interval=None) / 100.0
            mem_info = psutil.virtual_memory()
            mem_usage = mem_info.percent / 100.0
            
            # Battery info (if available)
            battery_level = 1.0
            try:
                battery = psutil.sensors_battery()
                if battery:
                    battery_level = battery.percent / 100.0
            except Exception:
                pass
            
            # Thermal info (simplified)
            thermal_throttling = 0.0
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Check if any sensor is above 80C
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current > 80:
                                thermal_throttling = 1.0
                                break
            except Exception:
                pass
            
            # Update state
            with self.lock:
                self.state = {
                    'cpu_util': cpu_util,
                    'mem_usage': mem_usage,
                    'battery_level': battery_level,
                    'thermal_throttling': thermal_throttling,
                    'timestamp': time.time()
                }
            
            time.sleep(self.poll_interval)
            
    def get_state(self) -> Dict:
        """Get current resource state."""
        with self.lock:
            return self.state.copy()
    
    def get_cpu_utilization(self) -> float:
        """Get current CPU utilization."""
        with self.lock:
            return self.state['cpu_util']
    
    def get_memory_usage(self) -> float:
        """Get current memory usage."""
        with self.lock:
            return self.state['mem_usage']
