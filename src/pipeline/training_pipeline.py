"""
VIGIA Medical AI - Training Pipeline
===================================

Automated retraining pipeline using existing Celery infrastructure.
Supports MONAI, YOLOv5, and MedGemma model training with medical validation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from ..core.celery_config import celery_app, configure_task_defaults, CELERY_AVAILABLE
from ..db.training_database import training_db, NPUAPGrade
from ..ml.model_tracking import model_tracker
from ..utils.audit_service import AuditService
from ..utils.secure_logger import SecureLogger

logger = SecureLogger(__name__)

class TrainingPipeline:
    """Automated training pipeline for medical AI models"""
    
    def __init__(self):
        self.audit_service = AuditService()
        self.training_configs = {
            'MONAI': {
                'epochs': 100,
                'batch_size': 8,
                'learning_rate': 1e-4,
                'early_stopping_patience': 10,
                'validation_split': 0.2,
                'min_dataset_size': 1000,
                'gpu_required': True
            },
            'YOLOv5': {
                'epochs': 300,
                'batch_size': 16,
                'learning_rate': 1e-3,
                'early_stopping_patience': 30,
                'validation_split': 0.15,
                'min_dataset_size': 500,
                'gpu_required': True
            },
            'MedGemma': {
                'epochs': 3,
                'batch_size': 4,
                'learning_rate': 2e-5,
                'early_stopping_patience': 1,
                'validation_split': 0.1,
                'min_dataset_size': 100,
                'gpu_required': True
            }
        }
        
        # Performance thresholds for triggering retraining
        self.retraining_thresholds = {
            'accuracy_drop': 0.05,  # 5% accuracy drop
            'safety_score_drop': 0.10,  # 10% safety score drop
            'false_negative_increase': 0.05,  # 5% increase in false negatives
            'days_since_training': 30,  # Retrain every 30 days minimum
            'new_data_threshold': 200  # 200+ new images triggers retraining
        }
        
        logger.info("TrainingPipeline initialized")
    
    async def initialize(self) -> None:
        """Initialize training pipeline"""
        await training_db.initialize()
        await model_tracker.initialize()
        logger.info("TrainingPipeline initialized successfully")
    
    def schedule_model_training(self,
                               model_type: str,
                               trigger_reason: str,
                               priority: str = 'normal',
                               custom_config: Dict[str, Any] = None) -> Optional[str]:
        """Schedule model training as Celery task"""
        
        if model_type not in self.training_configs:
            logger.error(f"Unsupported model type: {model_type}")
            return None
        
        # Generate training session ID
        session_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare training configuration
        base_config = self.training_configs[model_type].copy()
        if custom_config:
            base_config.update(custom_config)
        
        training_data = {
            'session_id': session_id,
            'model_type': model_type,
            'trigger_reason': trigger_reason,
            'config': base_config,
            'priority': priority,
            'scheduled_at': datetime.now().isoformat()
        }
        
        # Schedule task with appropriate priority
        queue_name = 'medical_priority' if priority in ['high', 'urgent'] else 'image_processing'
        
        try:
            # Schedule the training task
            task = train_medical_model_task.apply_async(
                args=[training_data],
                queue=queue_name,
                **configure_task_defaults('agent_coordination_task')
            )
            
            logger.info(f"Scheduled {model_type} training: {session_id} (Task: {task.id})")
            
            # Log to audit service
            asyncio.create_task(self.audit_service.log_activity(
                activity_type="training_scheduled",
                details={
                    'session_id': session_id,
                    'model_type': model_type,
                    'trigger_reason': trigger_reason,
                    'task_id': task.id
                }
            ))
            
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to schedule training: {e}")
            return None
    
    async def check_retraining_triggers(self) -> List[Dict[str, Any]]:
        """Check if any models need retraining"""
        
        triggers = []
        active_models = await training_db.get_active_model_versions()
        
        for model in active_models:
            model_triggers = await self._evaluate_model_triggers(model)
            if model_triggers:
                triggers.extend(model_triggers)
        
        return triggers
    
    async def _evaluate_model_triggers(self, model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate retraining triggers for a specific model"""
        
        triggers = []
        model_version = model['version_name']
        model_type = model['model_type']
        
        # Get recent performance metrics
        dashboard_metrics = await model_tracker.get_model_dashboard_metrics(model_version)
        
        # 1. Check accuracy drop
        if dashboard_metrics.get('daily_accuracy', 1.0) < model.get('overall_accuracy', 0.0) - self.retraining_thresholds['accuracy_drop']:
            triggers.append({
                'model_version': model_version,
                'model_type': model_type,
                'trigger_type': 'accuracy_drop',
                'priority': 'high',
                'details': {
                    'previous_accuracy': model.get('overall_accuracy'),
                    'current_accuracy': dashboard_metrics.get('daily_accuracy'),
                    'drop': model.get('overall_accuracy', 0.0) - dashboard_metrics.get('daily_accuracy', 1.0)
                }
            })
        
        # 2. Check training age
        deployed_at = model.get('deployed_at')
        if deployed_at:
            days_since_deployment = (datetime.now() - deployed_at).days
            if days_since_deployment > self.retraining_thresholds['days_since_training']:
                triggers.append({
                    'model_version': model_version,
                    'model_type': model_type,
                    'trigger_type': 'scheduled_retrain',
                    'priority': 'normal',
                    'details': {
                        'days_since_deployment': days_since_deployment,
                        'threshold': self.retraining_thresholds['days_since_training']
                    }
                })
        
        # 3. Check new data availability
        new_data_count = await self._count_new_training_data(deployed_at)
        if new_data_count > self.retraining_thresholds['new_data_threshold']:
            triggers.append({
                'model_version': model_version,
                'model_type': model_type,
                'trigger_type': 'new_data_available',
                'priority': 'normal',
                'details': {
                    'new_images': new_data_count,
                    'threshold': self.retraining_thresholds['new_data_threshold']
                }
            })
        
        return triggers
    
    async def _count_new_training_data(self, since_date: datetime = None) -> int:
        """Count new training data since deployment"""
        
        try:
            # Get dataset statistics
            if hasattr(training_db, '_mock_mode') and training_db._mock_mode:
                # Mock data for development
                return 50 if since_date and (datetime.now() - since_date).days > 7 else 0
            
            # Real database query would go here
            # For now, returning mock data
            return 0
            
        except Exception as e:
            logger.error(f"Failed to count new training data: {e}")
            return 0
    
    async def execute_automatic_retraining(self) -> Dict[str, Any]:
        """Execute automatic retraining based on triggers"""
        
        triggers = await self.check_retraining_triggers()
        scheduled_sessions = []
        
        for trigger in triggers:
            # Avoid duplicate training sessions
            if not await self._is_training_in_progress(trigger['model_type']):
                session_id = self.schedule_model_training(
                    model_type=trigger['model_type'],
                    trigger_reason=trigger['trigger_type'],
                    priority=trigger['priority']
                )
                
                if session_id:
                    scheduled_sessions.append({
                        'session_id': session_id,
                        'model_type': trigger['model_type'],
                        'trigger': trigger
                    })
        
        result = {
            'triggers_evaluated': len(triggers),
            'sessions_scheduled': len(scheduled_sessions),
            'scheduled_sessions': scheduled_sessions,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Automatic retraining: {len(scheduled_sessions)} sessions scheduled from {len(triggers)} triggers")
        
        return result
    
    async def _is_training_in_progress(self, model_type: str) -> bool:
        """Check if training is already in progress for model type"""
        
        # This would check active training sessions in database
        # For now, returning False to allow training
        return False
    
    def get_training_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get training status for a session"""
        
        # Check Celery task status
        # This would integrate with the actual training task
        return {
            'session_id': session_id,
            'status': 'running',
            'progress': 0.0,
            'estimated_completion': None
        }


# Celery Tasks for Training Pipeline
@celery_app.task(bind=True, **configure_task_defaults('agent_coordination_task'))
def train_medical_model_task(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
    """Celery task for training medical models"""
    
    try:
        session_id = training_data['session_id']
        model_type = training_data['model_type']
        config = training_data['config']
        
        logger.info(f"Starting training task: {session_id} ({model_type})")
        
        # Update progress
        self.update_state(state='TRAINING', meta={
            'session_id': session_id,
            'progress': 0,
            'stage': 'initialization'
        })
        
        # Execute training based on model type
        if model_type == 'MONAI':
            result = _train_monai_model(session_id, config, self)
        elif model_type == 'YOLOv5':
            result = _train_yolo_model(session_id, config, self)
        elif model_type == 'MedGemma':
            result = _train_medgemma_model(session_id, config, self)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"Training completed: {session_id} - Accuracy: {result.get('final_accuracy', 'N/A')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Training task failed: {e}")
        
        self.update_state(state='FAILURE', meta={
            'session_id': training_data.get('session_id'),
            'error': str(e),
            'stage': 'failed'
        })
        
        raise


def _train_monai_model(session_id: str, config: Dict[str, Any], task_context) -> Dict[str, Any]:
    """Train MONAI model for medical image analysis"""
    
    # Mock training for development
    import time
    
    # Simulate training stages
    stages = [
        ('data_loading', 0.1),
        ('preprocessing', 0.2),
        ('model_setup', 0.3),
        ('training', 0.8),
        ('validation', 0.9),
        ('evaluation', 1.0)
    ]
    
    for stage, progress in stages:
        time.sleep(2)  # Simulate work
        task_context.update_state(state='TRAINING', meta={
            'session_id': session_id,
            'progress': progress,
            'stage': stage,
            'current_epoch': int(progress * config['epochs']) if stage == 'training' else None
        })
    
    # Mock results
    return {
        'session_id': session_id,
        'model_type': 'MONAI',
        'final_accuracy': 0.93,
        'training_time_hours': 2.5,
        'epochs_completed': config['epochs'],
        'best_epoch': 85,
        'validation_accuracy': 0.91,
        'model_path': f'/models/monai_{session_id}.pth',
        'status': 'completed'
    }


def _train_yolo_model(session_id: str, config: Dict[str, Any], task_context) -> Dict[str, Any]:
    """Train YOLOv5 model for object detection"""
    
    # Mock training for development
    import time
    
    for i in range(10):
        time.sleep(1)
        progress = i / 10
        task_context.update_state(state='TRAINING', meta={
            'session_id': session_id,
            'progress': progress,
            'stage': 'training',
            'current_epoch': int(progress * config['epochs'])
        })
    
    return {
        'session_id': session_id,
        'model_type': 'YOLOv5',
        'final_accuracy': 0.89,
        'training_time_hours': 4.2,
        'epochs_completed': config['epochs'],
        'best_epoch': 220,
        'validation_accuracy': 0.87,
        'model_path': f'/models/yolo_{session_id}.pt',
        'status': 'completed'
    }


def _train_medgemma_model(session_id: str, config: Dict[str, Any], task_context) -> Dict[str, Any]:
    """Fine-tune MedGemma model"""
    
    # Mock training for development
    import time
    
    for i in range(5):
        time.sleep(3)
        progress = i / 5
        task_context.update_state(state='TRAINING', meta={
            'session_id': session_id,
            'progress': progress,
            'stage': 'fine_tuning',
            'current_epoch': int(progress * config['epochs'])
        })
    
    return {
        'session_id': session_id,
        'model_type': 'MedGemma',
        'final_accuracy': 0.95,
        'training_time_hours': 1.8,
        'epochs_completed': config['epochs'],
        'best_epoch': 2,
        'validation_accuracy': 0.94,
        'model_path': f'/models/medgemma_{session_id}',
        'status': 'completed'
    }


@celery_app.task(**configure_task_defaults('audit_log_task'))
def evaluate_model_task(model_version: str, test_dataset_config: Dict[str, Any]) -> Dict[str, Any]:
    """Celery task for model evaluation"""
    
    try:
        logger.info(f"Starting model evaluation: {model_version}")
        
        # This would load test dataset and run evaluation
        # For now, returning mock results
        
        results = {
            'model_version': model_version,
            'evaluation_timestamp': datetime.now().isoformat(),
            'overall_accuracy': 0.92,
            'test_dataset_size': 500,
            'status': 'completed'
        }
        
        logger.info(f"Model evaluation completed: {model_version}")
        return results
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise


@celery_app.task(**configure_task_defaults('audit_log_task'))
def check_retraining_triggers_task() -> Dict[str, Any]:
    """Periodic task to check retraining triggers"""
    
    try:
        # This would be called by Celery Beat for periodic checks
        pipeline = TrainingPipeline()
        
        # Run async function in sync context
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(pipeline.execute_automatic_retraining())
        loop.close()
        
        return result
        
    except Exception as e:
        logger.error(f"Retraining trigger check failed: {e}")
        raise


# Singleton instance
training_pipeline = TrainingPipeline()

__all__ = [
    'TrainingPipeline', 
    'training_pipeline',
    'train_medical_model_task',
    'evaluate_model_task',
    'check_retraining_triggers_task'
]