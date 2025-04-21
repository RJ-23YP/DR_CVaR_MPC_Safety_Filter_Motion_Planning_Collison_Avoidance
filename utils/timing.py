"""
Timing and profiling utilities for the DR-CVaR safety filtering.
"""
import time
import functools
import numpy as np

class Timer:
    """
    Simple timer for measuring execution time.
    """
    def __init__(self, name=None):
        self.name = name
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.elapsed = time.time() - self.start_time
        self.start_time = None
        return self.elapsed
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.stop()
        if self.name:
            print(f"{self.name}: {self.elapsed:.6f} seconds")

def timeit(func):
    """
    Decorator to measure function execution time.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timer_name = f"{func.__name__}"
        with Timer(timer_name):
            result = func(*args, **kwargs)
        return result
    return wrapper

class TimingStats:
    """
    Class for collecting and analyzing timing statistics.
    """
    def __init__(self):
        self.data = {}
    
    def add(self, name, time_value):
        """Add a timing measurement."""
        if name not in self.data:
            self.data[name] = []
        self.data[name].append(time_value)
    
    def get_stats(self, name):
        """Get statistics for a particular timing measurement."""
        if name not in self.data or not self.data[name]:
            return None
        
        times = self.data[name]
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'count': len(times)
        }
    
    def print_stats(self):
        """Print statistics for all timing measurements."""
        for name, times in self.data.items():
            stats = self.get_stats(name)
            print(f"{name}:")
            print(f"  Mean: {stats['mean']:.6f} seconds")
            print(f"  Std:  {stats['std']:.6f} seconds")
            print(f"  Min:  {stats['min']:.6f} seconds")
            print(f"  Max:  {stats['max']:.6f} seconds")
            print(f"  Count: {stats['count']}")