"""
Configuration Manager
Loads and manages all system parameters from YAML files
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
from log import log_with_timestamp


class ConfigManager:
    """
    Configuration Manager - Loads settings from YAML files
    
    Features:
    - Load configuration from YAML files
    - Environment variable support
    - Default values
    - Validation
    - Hot reload (optional)
    """
    
    def __init__(self, config_path: str = "config_webcam.yaml"):
        """
        Args:
            config_path: Configuration file path
        """
        self.config_path = config_path
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
                log_with_timestamp(f"Configuration loaded: {self.config_path}", "CONFIG")
            else:
                log_with_timestamp(f"Configuration file not found: {self.config_path}", "WARNING")
                log_with_timestamp("Using default values...", "CONFIG")
                self._set_defaults()
        except Exception as e:
            log_with_timestamp(f"Configuration could not be loaded: {e}", "ERROR")
            log_with_timestamp("Using default values...", "CONFIG")
            self._set_defaults()
    
    def _set_defaults(self):
        """Default configuration values"""
        self.config = {
            'camera': {
                'source': 0,
                'resolution': [640, 480],
                'buffer_size': 1
            },
            'yolo': {
                'model_path': 'yolo11n-pose.pt',
                'device': 'auto'
            },
            'tracking': {
                'max_history_length': 30,
                'use_norfair': True,
                'distance_function': 'keypoint',
                'distance_threshold': 0.8,
                'hit_counter_max': 60,
                'initialization_delay': 2,
                'pointwise_hit_counter_max': 4,
                'use_reid': True,
                'reid_distance_threshold': 0.3,
                'reid_hit_counter_max': 150,
                'keypoint_weight': 0.6,
                'reid_weight': 0.4,
                'use_persistent_reid': True,
                'persistent_db_path': 'person_database.json',
                'persistent_db_type': 'json',
                'persistent_similarity_threshold': 0.65,
                'max_persons': 1000,
                'auto_save': True
            },
            'performance': {
                'track_every_n_frames': 1
            },
            'ui': {
                'window_name': 'YOLO11 Pose Detection'
            },
            'logging': {
                'log_to_file': True,
                'log_file': 'logs/yolo_detection.log'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Dot-separated key (e.g., 'camera.fps')
            default: Default value
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value
        
        Args:
            key: Dot-separated key
            value: New value
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the last key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
    
    def get_camera_config(self) -> Dict[str, Any]:
        """Get camera configuration"""
        return self.get('camera', {})
    
    def get_yolo_config(self) -> Dict[str, Any]:
        """Get YOLO configuration"""
        return self.get('yolo', {})
    
    def get_tracking_config(self) -> Dict[str, Any]:
        """Get tracking configuration"""
        return self.get('tracking', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.get('performance', {})
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration"""
        return self.get('ui', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get('logging', {})
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection configuration"""
        return self.get('detection', {})
    
    def get_pose_measurement_config(self) -> Dict[str, Any]:
        """Get pose measurement configuration"""
        return self.get('pose_measurement', {})
    
    def get_skeletal_config(self) -> Dict[str, Any]:
        """Get skeletal biometrics configuration"""
        return self.get('skeletal', {})
    
    def get_rtsp_config(self) -> Dict[str, Any]:
        """Get RTSP configuration"""
        return self.get('rtsp', {})
    
    def save_config(self, path: Optional[str] = None):
        """
        Save configuration to file
        
        Args:
            path: Save path (use current path if None)
        """
        save_path = path or self.config_path
        
        try:
            # Create directory
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False)
            log_with_timestamp(f"Configuration saved: {save_path}", "CONFIG")
        except Exception as e:
            log_with_timestamp(f"Configuration could not be saved: {e}", "ERROR")
    
    def reload_config(self):
        """Reload configuration"""
        log_with_timestamp("Reloading configuration...", "CONFIG")
        self._load_config()
    
    def validate_config(self) -> bool:
        """
        Validate configuration
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            required_sections = ['camera', 'yolo', 'tracking', 'ui']
            
            for section in required_sections:
                if section not in self.config:
                    log_with_timestamp(f"Missing section: {section}", "ERROR")
                    return False
            
            # Camera validation
            camera = self.get_camera_config()
            if camera.get('resolution', [0, 0])[0] <= 0:
                log_with_timestamp("Camera resolution must be positive", "ERROR")
                return False
            
            # YOLO validation
            yolo = self.get_yolo_config()
            if not os.path.exists(yolo.get('model_path', '')):
                log_with_timestamp(f"YOLO model file not found: {yolo.get('model_path')}", "WARNING")
            
            # Tracking validation
            tracking = self.get_tracking_config()
            if tracking.get('keypoint_weight', 0) + tracking.get('reid_weight', 0) != 1.0:
                log_with_timestamp("Keypoint and ReID weights should sum to 1.0", "WARNING")
            
            log_with_timestamp("Configuration validation successful", "CONFIG")
            return True
            
        except Exception as e:
            log_with_timestamp(f"Configuration validation error: {e}", "ERROR")
            return False
    
    def print_config(self):
        """Print configuration"""
        log_with_timestamp("Current Configuration:", "CONFIG")
        log_with_timestamp("=" * 50, "CONFIG")
        
        for section, values in self.config.items():
            log_with_timestamp(f"\n[{section.upper()}]", "CONFIG")
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, dict):
                        log_with_timestamp(f"  {key}:", "CONFIG")
                        for sub_key, sub_value in value.items():
                            log_with_timestamp(f"    {sub_key}: {sub_value}", "CONFIG")
                    else:
                        log_with_timestamp(f"  {key}: {value}", "CONFIG")
            else:
                log_with_timestamp(f"  {values}", "CONFIG")
    
    def get_device(self) -> str:
        """Get YOLO device (with auto detection)"""
        device = self.get('yolo.device', 'auto')
        
        if device == 'auto':
            try:
                import torch
                if torch.cuda.is_available():
                    device = 'cuda'
                    log_with_timestamp("GPU found, using CUDA", "SYSTEM")
                else:
                    device = 'cpu'
                    log_with_timestamp("GPU not found, using CPU", "WARNING")
            except ImportError:
                device = 'cpu'
                log_with_timestamp("PyTorch not found, using CPU", "WARNING")
        
        return device
    
    def create_directories(self):
        """Create required directories"""
        directories = [
            self.get('environment.data_dir', 'data'),
            self.get('environment.models_dir', 'models'),
            self.get('environment.logs_dir', 'logs'),
            self.get('environment.temp_dir', 'temp'),
            self.get('advanced.debug_output_dir', 'debug_output')
        ]
        
        for directory in directories:
            if directory:
                os.makedirs(directory, exist_ok=True)
                log_with_timestamp(f"Directory ready: {directory}", "SYSTEM")


# Global config instance
config = ConfigManager()


def get_config(config_file: str = "config_webcam.yaml") -> ConfigManager:
    """Get config instance - load from specified file"""
    if config_file == "config_webcam.yaml":
        return config
    else:
        # Create new config instance
        return ConfigManager(config_file)


def reload_config():
    """Reload global config"""
    global config
    config.reload_config()


# Convenience functions
def get_camera_config() -> Dict[str, Any]:
    """Get camera configuration"""
    return config.get_camera_config()


def get_yolo_config() -> Dict[str, Any]:
    """Get YOLO configuration"""
    return config.get_yolo_config()


def get_tracking_config() -> Dict[str, Any]:
    """Get tracking configuration"""
    return config.get_tracking_config()


def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration"""
    return config.get_performance_config()


def get_ui_config() -> Dict[str, Any]:
    """Get UI configuration"""
    return config.get_ui_config()


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration"""
    return config.get_logging_config()


def get_detection_config() -> Dict[str, Any]:
    """Get detection configuration"""
    return config.get_detection_config()


def get_pose_measurement_config() -> Dict[str, Any]:
    """Get pose measurement configuration"""
    return config.get_pose_measurement_config()


def get_skeletal_config() -> Dict[str, Any]:
    """Get skeletal biometrics configuration"""
    return config.get_skeletal_config()


def get_rtsp_config() -> Dict[str, Any]:
    """Get RTSP configuration"""
    return config.get_rtsp_config()


if __name__ == "__main__":
    # Test configuration
    config_manager = ConfigManager()
    
    print("[TEST] Configuration Test:")
    print(f"Camera FPS: {config_manager.get('camera.fps')}")
    print(f"YOLO Model: {config_manager.get('yolo.model_path')}")
    print(f"Tracking Threshold: {config_manager.get('tracking.distance_threshold')}")
    print(f"Device: {config_manager.get_device()}")
    
    # Validation
    config_manager.validate_config()
    
    # Print full config
    config_manager.print_config()