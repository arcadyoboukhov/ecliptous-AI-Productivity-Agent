"""
System Resource Monitoring

Monitors CPU, RAM, GPU, and Disk I/O usage.
Provides normalized metrics (0.0 to 1.0 or percentage).
"""

import threading
import time
from typing import Dict, Optional
from datetime import datetime, timezone

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False


class SystemResourceMonitor:
    """
    Monitors system resource usage.
    
    Signals collected:
    - CPU usage percentage (0-100)
    - RAM usage percentage (0-100)
    - GPU usage percentage (0-100) - if available
    - Disk I/O rate (MB/s)
    """
    
    def __init__(self, poll_interval: float = 2.0):
        self.poll_interval = poll_interval
        self.is_monitoring = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
        # Current metrics
        self._cpu_percent = 0.0
        self._ram_percent = 0.0
        self._gpu_percent = 0.0
        self._disk_read_mbps = 0.0
        self._disk_write_mbps = 0.0
        self._last_update = None
        
        # Disk I/O tracking
        self._last_disk_io = None
        self._last_disk_time = None
    
    def start(self):
        """Start monitoring system resources."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._stop_event.clear()
        
        # Initialize psutil CPU tracking (first call returns 0)
        if HAS_PSUTIL:
            psutil.cpu_percent(interval=None)
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self):
        """Stop monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
    
    def get_metrics(self) -> Dict:
        """
        Get current resource metrics.
        
        Returns:
            dict with keys:
                - cpu_percent: float (0-100)
                - ram_percent: float (0-100)
                - gpu_percent: float (0-100, or None if unavailable)
                - disk_read_mbps: float (MB/s)
                - disk_write_mbps: float (MB/s)
                - last_update: ISO timestamp
        """
        return {
            "cpu_percent": self._cpu_percent,
            "ram_percent": self._ram_percent,
            "gpu_percent": self._gpu_percent if HAS_GPUTIL else None,
            "disk_read_mbps": self._disk_read_mbps,
            "disk_write_mbps": self._disk_write_mbps,
            "last_update": self._last_update.isoformat() if self._last_update else None
        }
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._update_cpu()
                self._update_ram()
                self._update_gpu()
                self._update_disk_io()
                self._last_update = datetime.now(timezone.utc)
            except Exception:
                pass
            
            self._stop_event.wait(self.poll_interval)
    
    def _update_cpu(self):
        """Update CPU usage percentage."""
        if not HAS_PSUTIL:
            return
        
        try:
            # Get CPU percentage (non-blocking)
            self._cpu_percent = psutil.cpu_percent(interval=None)
        except Exception:
            pass
    
    def _update_ram(self):
        """Update RAM usage percentage."""
        if not HAS_PSUTIL:
            return
        
        try:
            mem = psutil.virtual_memory()
            self._ram_percent = mem.percent
        except Exception:
            pass
    
    def _update_gpu(self):
        """Update GPU usage percentage."""
        if not HAS_GPUTIL:
            return
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                # Use first GPU or average if multiple
                if len(gpus) == 1:
                    self._gpu_percent = gpus[0].load * 100
                else:
                    avg_load = sum(gpu.load for gpu in gpus) / len(gpus)
                    self._gpu_percent = avg_load * 100
        except Exception:
            pass
    
    def _update_disk_io(self):
        """Update disk I/O rates (MB/s)."""
        if not HAS_PSUTIL:
            return
        
        try:
            current_io = psutil.disk_io_counters()
            current_time = time.time()
            
            if self._last_disk_io is not None and self._last_disk_time is not None:
                # Calculate rates
                time_delta = current_time - self._last_disk_time
                
                if time_delta > 0:
                    read_bytes = current_io.read_bytes - self._last_disk_io.read_bytes
                    write_bytes = current_io.write_bytes - self._last_disk_io.write_bytes
                    
                    self._disk_read_mbps = (read_bytes / time_delta) / (1024 * 1024)
                    self._disk_write_mbps = (write_bytes / time_delta) / (1024 * 1024)
            
            self._last_disk_io = current_io
            self._last_disk_time = current_time
            
        except Exception:
            pass


# Global singleton
_resource_monitor = None


def get_resource_monitor() -> SystemResourceMonitor:
    """Get or create the global resource monitor singleton."""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = SystemResourceMonitor()
    return _resource_monitor


def get_system_metrics() -> Dict:
    """
    Get current system resource metrics.
    
    Returns dict with cpu_percent, ram_percent, gpu_percent, disk I/O rates.
    """
    monitor = get_resource_monitor()
    if not monitor.is_monitoring:
        monitor.start()
    return monitor.get_metrics()


if __name__ == "__main__":
    print("Testing System Resource Monitor...")
    print(f"psutil available: {HAS_PSUTIL}")
    print(f"GPUtil available: {HAS_GPUTIL}")
    print()
    
    monitor = get_resource_monitor()
    monitor.start()
    
    print("Monitoring for 10 seconds...")
    for i in range(10):
        time.sleep(1)
        metrics = monitor.get_metrics()
        gpu_str = f"{metrics['gpu_percent']:.1f}%" if metrics['gpu_percent'] is not None else "N/A"
        print(f"\r[{i+1}/10] CPU: {metrics['cpu_percent']:.1f}% | "
              f"RAM: {metrics['ram_percent']:.1f}% | "
              f"GPU: {gpu_str} | "
              f"Disk R: {metrics['disk_read_mbps']:.2f} MB/s | "
              f"Disk W: {metrics['disk_write_mbps']:.2f} MB/s", end='')
    
    print("\n\nFinal metrics:")
    print(metrics)
    
    monitor.stop()
