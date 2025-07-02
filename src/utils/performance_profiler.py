"""
Performance profiling utilities for Vigia using py-spy
Provides decorators and utilities for profiling code execution
"""

import os
import time
import logging
import subprocess
from functools import wraps
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Performance profiler using py-spy"""
    
    def __init__(self, 
                 output_dir: Optional[Path] = None,
                 enabled: bool = True):
        """
        Initialize performance profiler
        
        Args:
            output_dir: Directory to save profiling data
            enabled: Whether profiling is enabled
        """
        self.output_dir = output_dir or Path("logs/profiles")
        self.enabled = enabled and self._check_pyspy_available() and os.getenv("ENABLE_PYSPY", "false").lower() == "true"
        
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Performance profiling enabled")
    
    def _check_pyspy_available(self) -> bool:
        """Check if py-spy is available"""
        try:
            result = subprocess.run(["py-spy", "--version"], 
                                  capture_output=True, 
                                  text=True, 
                                  check=False)
            return result.returncode == 0
        except FileNotFoundError:
            logger.debug("py-spy not found. Performance profiling disabled.")
            return False
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Any:
        """
        Profile a function execution
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        if not self.enabled:
            return func(*args, **kwargs)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        func_name = func.__name__
        output_file = self.output_dir / f"profile_{func_name}_{timestamp}.svg"
        
        # Create a wrapper script to run the function
        script_content = f"""
import sys
sys.path.append('{Path.cwd()}')
from {func.__module__} import {func.__name__}

# Run the function
result = {func.__name__}(*{args}, **{kwargs})
"""
        
        script_path = self.output_dir / f"temp_profile_{timestamp}.py"
        script_path.write_text(script_content)
        
        try:
            # Run py-spy
            cmd = [
                "py-spy", "record",
                "-o", str(output_file),
                "-d", "30",  # Duration in seconds
                "--", "python", str(script_path)
            ]
            
            logger.info(f"Starting profiling for {func_name}")
            subprocess.run(cmd, check=True)
            logger.info(f"Profile saved to: {output_file}")
            
            # Clean up
            script_path.unlink()
            
            # Run the function normally to return result
            return func(*args, **kwargs)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Profiling failed: {e}")
            # Clean up and run function normally
            if script_path.exists():
                script_path.unlink()
            return func(*args, **kwargs)
    
    def profile(self, duration: int = 30, output_format: str = "flamegraph"):
        """
        Decorator to profile a function
        
        Args:
            duration: Profile duration in seconds
            output_format: Output format (flamegraph, speedscope, raw)
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                func_name = func.__name__
                
                # Determine file extension based on format
                ext_map = {
                    "flamegraph": "svg",
                    "speedscope": "json",
                    "raw": "txt"
                }
                ext = ext_map.get(output_format, "svg")
                output_file = self.output_dir / f"profile_{func_name}_{timestamp}.{ext}"
                
                # Start profiling in background
                import os
                pid = os.getpid()
                
                cmd = [
                    "py-spy", "record",
                    "-o", str(output_file),
                    "-d", str(duration),
                    "-p", str(pid)
                ]
                
                if output_format == "speedscope":
                    cmd.extend(["-f", "speedscope"])
                elif output_format == "raw":
                    cmd.extend(["-f", "raw"])
                
                # Start py-spy in background
                process = subprocess.Popen(cmd)
                
                try:
                    # Run the function
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    
                    # Wait for profiling to complete if function finished quickly
                    if elapsed < duration:
                        time.sleep(0.5)  # Give py-spy time to finish
                    
                    logger.info(f"Profile saved to: {output_file}")
                    return result
                    
                finally:
                    # Ensure py-spy is terminated
                    try:
                        process.terminate()
                        process.wait(timeout=5)
                    except:
                        pass
                        
            return wrapper
        return decorator


# Global instance
profiler = PerformanceProfiler()


def profile_performance(duration: int = 30, output_format: str = "flamegraph"):
    """
    Decorator to profile function performance
    
    Usage:
        @profile_performance(duration=60)
        def expensive_operation():
            # Your code here
            pass
    """
    return profiler.profile(duration, output_format)


def get_profile_summary(profiles_dir: Path = Path("logs/profiles")) -> Dict[str, Any]:
    """
    Get summary of profiling data
    
    Returns:
        Summary of profiling sessions
    """
    if not profiles_dir.exists():
        return {"error": "No profiling data found"}
    
    profiles = []
    for profile_file in profiles_dir.glob("profile_*.svg"):
        # Extract metadata from filename
        parts = profile_file.stem.split("_")
        if len(parts) >= 3:
            func_name = "_".join(parts[1:-2])
            timestamp = f"{parts[-2]}_{parts[-1]}"
            
            profiles.append({
                "function": func_name,
                "timestamp": timestamp,
                "file": str(profile_file),
                "size_kb": profile_file.stat().st_size / 1024
            })
    
    return {
        "total_profiles": len(profiles),
        "profiles": sorted(profiles, key=lambda x: x["timestamp"], reverse=True),
        "total_size_mb": sum(p["size_kb"] for p in profiles) / 1024
    }


# Utility function for manual profiling
def start_profiling(output_file: Optional[str] = None, duration: int = 30) -> Optional[subprocess.Popen]:
    """
    Start profiling the current process
    
    Args:
        output_file: Output file path
        duration: Profile duration in seconds
        
    Returns:
        Subprocess handle or None
    """
    if not profiler.enabled:
        return None
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = profiler.output_dir / f"profile_manual_{timestamp}.svg"
    
    pid = os.getpid()
    cmd = [
        "py-spy", "record",
        "-o", str(output_file),
        "-d", str(duration),
        "-p", str(pid)
    ]
    
    try:
        process = subprocess.Popen(cmd)
        logger.info(f"Started profiling PID {pid} for {duration} seconds")
        return process
    except Exception as e:
        logger.error(f"Failed to start profiling: {e}")
        return None