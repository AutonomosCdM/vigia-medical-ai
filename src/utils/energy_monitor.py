"""
Energy consumption monitoring for Vigia using CodeCarbon
Tracks CO2 emissions and energy usage during model inference
"""

import os
import logging
from functools import wraps
from typing import Optional, Dict, Any, Callable
from pathlib import Path

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    EmissionsTracker = None

logger = logging.getLogger(__name__)


class EnergyMonitor:
    """Monitor energy consumption and CO2 emissions"""
    
    def __init__(self, 
                 project_name: str = "vigia-medical-detection",
                 output_dir: Optional[Path] = None,
                 enabled: bool = True):
        """
        Initialize energy monitor
        
        Args:
            project_name: Name for tracking
            output_dir: Directory to save emissions data
            enabled: Whether monitoring is enabled
        """
        self.project_name = project_name
        self.output_dir = output_dir or Path("logs/emissions")
        self.enabled = enabled and CODECARBON_AVAILABLE and os.getenv("ENABLE_CODECARBON", "false").lower() == "true"
        self.tracker: Optional[EmissionsTracker] = None
        
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Energy monitoring enabled for {project_name}")
        elif not CODECARBON_AVAILABLE:
            logger.debug("CodeCarbon not installed. Energy monitoring disabled.")
    
    def start_tracking(self, task_name: str = "inference") -> None:
        """Start tracking emissions for a task"""
        if not self.enabled:
            return
            
        try:
            self.tracker = EmissionsTracker(
                project_name=self.project_name,
                measure_power_secs=1,
                output_dir=str(self.output_dir),
                save_to_file=True,
                save_to_api=False,
                log_level="ERROR",  # Reduce codecarbon verbosity
                gpu_ids=None,  # Track all available GPUs
            )
            self.tracker.start()
            logger.debug(f"Started energy tracking for task: {task_name}")
        except Exception as e:
            logger.warning(f"Failed to start energy tracking: {e}")
            self.enabled = False
    
    def stop_tracking(self) -> Optional[Dict[str, float]]:
        """
        Stop tracking and return emissions data
        
        Returns:
            Dictionary with emissions data or None
        """
        if not self.enabled or not self.tracker:
            return None
            
        try:
            emissions = self.tracker.stop()
            
            # Create summary
            summary = {
                "emissions_kg_co2": emissions,
                "emissions_g_co2": emissions * 1000,
                "duration_seconds": self.tracker.final_emissions_data.duration,
                "energy_consumed_kwh": self.tracker.final_emissions_data.energy_consumed,
                "energy_consumed_wh": self.tracker.final_emissions_data.energy_consumed * 1000,
                "ram_energy_kwh": self.tracker.final_emissions_data.ram_energy,
                "cpu_energy_kwh": self.tracker.final_emissions_data.cpu_energy,
                "gpu_energy_kwh": self.tracker.final_emissions_data.gpu_energy,
            }
            
            logger.info(f"Energy tracking completed: {emissions:.6f} kg CO2 emitted")
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to stop energy tracking: {e}")
            return None
        finally:
            self.tracker = None
    
    def track(self, task_name: str = "inference"):
        """Decorator to track energy consumption of a function"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.start_tracking(task_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    emissions_data = self.stop_tracking()
                    if emissions_data:
                        # Add emissions data to result if it's a dict
                        if isinstance(result, dict):
                            result["energy_metrics"] = emissions_data
            return wrapper
        return decorator


# Global instance
energy_monitor = EnergyMonitor()


def track_energy(task_name: str = "inference"):
    """
    Decorator to track energy consumption
    
    Usage:
        @track_energy("detection")
        def detect_injuries(image):
            # Your detection code
            pass
    """
    return energy_monitor.track(task_name)


def get_energy_summary(emissions_dir: Path = Path("logs/emissions")) -> Dict[str, Any]:
    """
    Get summary of all emissions from saved files
    
    Returns:
        Summary statistics of energy consumption
    """
    if not emissions_dir.exists():
        return {"error": "No emissions data found"}
    
    try:
        import pandas as pd
        
        # Find all emissions CSV files
        csv_files = list(emissions_dir.glob("emissions*.csv"))
        if not csv_files:
            return {"error": "No emissions CSV files found"}
        
        # Read and combine all data
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Calculate summary statistics
        summary = {
            "total_emissions_kg_co2": combined_df["emissions"].sum(),
            "total_energy_kwh": combined_df["energy_consumed"].sum(),
            "total_duration_hours": combined_df["duration"].sum() / 3600,
            "average_power_watts": (combined_df["energy_consumed"].sum() * 1000) / (combined_df["duration"].sum() / 3600),
            "total_runs": len(combined_df),
            "by_project": combined_df.groupby("project_name").agg({
                "emissions": "sum",
                "energy_consumed": "sum",
                "duration": "sum"
            }).to_dict()
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to generate energy summary: {e}")
        return {"error": str(e)}