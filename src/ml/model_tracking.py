"""
VIGIA Medical AI - Model Performance Tracking
============================================

Advanced ML model tracking with medical validation, drift detection,
and A/B testing for continuous improvement.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import aioredis
import hashlib

from ..db.training_database import training_db, NPUAPGrade
from ..utils.audit_service import AuditService

logger = logging.getLogger(__name__)

class ModelPerformanceTracker:
    """Comprehensive model performance tracking for medical AI"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or "redis://localhost:6379/3"
        self.redis_client = None
        self.audit_service = AuditService()
        
        # Performance thresholds for medical safety
        self.medical_thresholds = {
            'minimum_accuracy': 0.85,
            'critical_stage_accuracy': 0.90,  # Stages 3-4 require higher accuracy
            'drift_threshold': 0.05,  # 5% accuracy drop triggers alert
            'confidence_threshold': 0.80,
            'false_negative_threshold': 0.10,  # Max 10% false negatives for safety
        }
        
        logger.info("ModelPerformanceTracker initialized")
    
    async def initialize(self) -> None:
        """Initialize Redis connection for real-time metrics"""
        try:
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                decode_responses=True
            )
            logger.info("ModelPerformanceTracker Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed, using mock mode: {e}")
            self.redis_client = None
    
    async def track_model_inference(self,
                                    model_version: str,
                                    batman_token: str,
                                    prediction: Dict[str, Any],
                                    ground_truth: str = None,
                                    confidence_scores: Dict[str, float] = None,
                                    inference_time_ms: float = None) -> None:
        """Track individual model inference for performance monitoring"""
        
        timestamp = datetime.now()
        
        inference_data = {
            'model_version': model_version,
            'batman_token': batman_token,
            'predicted_grade': prediction.get('predicted_grade'),
            'ground_truth': ground_truth,
            'confidence_scores': confidence_scores or {},
            'max_confidence': max(confidence_scores.values()) if confidence_scores else None,
            'inference_time_ms': inference_time_ms,
            'timestamp': timestamp.isoformat(),
            'correct_prediction': prediction.get('predicted_grade') == ground_truth if ground_truth else None
        }
        
        # Store in Redis for real-time analytics
        if self.redis_client:
            try:
                # Real-time metrics
                await self._update_realtime_metrics(model_version, inference_data)
                
                # Store detailed inference (expire after 7 days)
                inference_key = f"vigia:inference:{model_version}:{timestamp.strftime('%Y%m%d_%H%M%S')}_{batman_token}"
                await self.redis_client.setex(
                    inference_key,
                    7 * 24 * 3600,  # 7 days
                    json.dumps(inference_data, default=str)
                )
                
            except Exception as e:
                logger.error(f"Failed to track inference in Redis: {e}")
        
        # Audit log for compliance
        await self.audit_service.log_activity(
            activity_type="model_inference",
            batman_token=batman_token,
            details={
                'model_version': model_version,
                'predicted_grade': prediction.get('predicted_grade'),
                'confidence': inference_data['max_confidence']
            }
        )
        
        logger.debug(f"Tracked inference: {model_version} -> {prediction.get('predicted_grade')} (confidence: {inference_data['max_confidence']})")
    
    async def evaluate_model_performance(self,
                                         model_version: str,
                                         test_dataset: List[Dict[str, Any]],
                                         model_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive model evaluation with medical-specific metrics"""
        
        if len(test_dataset) != len(model_predictions):
            raise ValueError("Test dataset and predictions must have same length")
        
        # Extract ground truth and predictions
        y_true = [item['npuap_grade'] for item in test_dataset]
        y_pred = [pred['predicted_grade'] for pred in model_predictions]
        confidence_scores = [pred.get('confidence_scores', {}) for pred in model_predictions]
        
        # Basic classification metrics
        cm = confusion_matrix(y_true, y_pred, labels=[g.value for g in NPUAPGrade])
        report = classification_report(y_true, y_pred, labels=[g.value for g in NPUAPGrade], output_dict=True)
        
        # Medical-specific metrics
        medical_metrics = await self._calculate_medical_metrics(y_true, y_pred, confidence_scores)
        
        # Stage-specific accuracy
        stage_accuracies = await self._calculate_stage_accuracies(y_true, y_pred)
        
        # Clinical validation metrics
        clinical_metrics = await self._calculate_clinical_metrics(y_true, y_pred, test_dataset)
        
        evaluation_results = {
            'model_version': model_version,
            'evaluation_timestamp': datetime.now().isoformat(),
            'dataset_size': len(test_dataset),
            'overall_accuracy': report['accuracy'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'macro_f1': report['macro avg']['f1-score'],
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'stage_accuracies': stage_accuracies,
            'medical_metrics': medical_metrics,
            'clinical_metrics': clinical_metrics,
            'safety_assessment': await self._assess_medical_safety(medical_metrics, stage_accuracies)
        }
        
        # Store results in database
        await self._store_evaluation_results(evaluation_results)
        
        # Check for performance degradation
        await self._check_performance_drift(model_version, evaluation_results)
        
        logger.info(f"Model evaluation completed: {model_version} - Accuracy: {report['accuracy']:.3f}")
        
        return evaluation_results
    
    async def _calculate_medical_metrics(self,
                                         y_true: List[str],
                                         y_pred: List[str],
                                         confidence_scores: List[Dict[str, float]]) -> Dict[str, Any]:
        """Calculate medical-specific performance metrics"""
        
        # Critical stage detection (stages 3-4)
        critical_stages = ['3', '4']
        critical_true = [1 if stage in critical_stages else 0 for stage in y_true]
        critical_pred = [1 if stage in critical_stages else 0 for stage in y_pred]
        
        # Safety metrics
        false_negatives = sum(1 for t, p in zip(y_true, y_pred) if t in critical_stages and p not in critical_stages)
        total_critical = sum(critical_true)
        false_negative_rate = false_negatives / total_critical if total_critical > 0 else 0
        
        # Confidence analysis
        avg_confidence = np.mean([max(scores.values()) if scores else 0.5 for scores in confidence_scores])
        low_confidence_predictions = sum(1 for scores in confidence_scores 
                                       if scores and max(scores.values()) < self.medical_thresholds['confidence_threshold'])
        
        # Grade progression accuracy (adjacent stages)
        adjacent_accuracy = self._calculate_adjacent_stage_accuracy(y_true, y_pred)
        
        return {
            'critical_stage_sensitivity': 1 - false_negative_rate,
            'false_negative_rate': false_negative_rate,
            'average_confidence': avg_confidence,
            'low_confidence_rate': low_confidence_predictions / len(confidence_scores),
            'adjacent_stage_accuracy': adjacent_accuracy,
            'safety_score': self._calculate_safety_score(false_negative_rate, avg_confidence)
        }
    
    async def _calculate_stage_accuracies(self,
                                          y_true: List[str],
                                          y_pred: List[str]) -> Dict[str, float]:
        """Calculate accuracy for each NPUAP stage"""
        
        stage_accuracies = {}
        
        for grade in NPUAPGrade:
            stage_value = grade.value
            stage_indices = [i for i, true_stage in enumerate(y_true) if true_stage == stage_value]
            
            if stage_indices:
                correct_predictions = sum(1 for i in stage_indices if y_pred[i] == stage_value)
                stage_accuracies[stage_value] = correct_predictions / len(stage_indices)
            else:
                stage_accuracies[stage_value] = 0.0
        
        return stage_accuracies
    
    async def _calculate_clinical_metrics(self,
                                          y_true: List[str],
                                          y_pred: List[str],
                                          test_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate clinically relevant metrics"""
        
        # Age-based performance analysis
        age_groups = {}
        for i, data in enumerate(test_dataset):
            age = data.get('patient_age', 'unknown')
            age_group = self._get_age_group(age)
            
            if age_group not in age_groups:
                age_groups[age_group] = {'correct': 0, 'total': 0}
            
            age_groups[age_group]['total'] += 1
            if y_true[i] == y_pred[i]:
                age_groups[age_group]['correct'] += 1
        
        age_performance = {
            group: data['correct'] / data['total'] if data['total'] > 0 else 0
            for group, data in age_groups.items()
        }
        
        # Body location performance
        location_performance = {}
        for i, data in enumerate(test_dataset):
            location = data.get('body_location', 'unknown')
            
            if location not in location_performance:
                location_performance[location] = {'correct': 0, 'total': 0}
            
            location_performance[location]['total'] += 1
            if y_true[i] == y_pred[i]:
                location_performance[location]['correct'] += 1
        
        location_accuracy = {
            loc: data['correct'] / data['total'] if data['total'] > 0 else 0
            for loc, data in location_performance.items()
        }
        
        return {
            'age_group_performance': age_performance,
            'body_location_performance': location_accuracy,
            'demographic_bias_score': self._calculate_bias_score(age_performance)
        }
    
    def _calculate_adjacent_stage_accuracy(self, y_true: List[str], y_pred: List[str]) -> float:
        """Calculate accuracy for predictions within ±1 stage"""
        
        stage_order = ['0', '1', '2', '3', '4', 'U', 'DTI']
        stage_to_idx = {stage: i for i, stage in enumerate(stage_order)}
        
        adjacent_correct = 0
        valid_comparisons = 0
        
        for true_stage, pred_stage in zip(y_true, y_pred):
            if true_stage in stage_to_idx and pred_stage in stage_to_idx:
                true_idx = stage_to_idx[true_stage]
                pred_idx = stage_to_idx[pred_stage]
                
                # Consider correct if within ±1 stage
                if abs(true_idx - pred_idx) <= 1:
                    adjacent_correct += 1
                
                valid_comparisons += 1
        
        return adjacent_correct / valid_comparisons if valid_comparisons > 0 else 0.0
    
    def _calculate_safety_score(self, false_negative_rate: float, avg_confidence: float) -> float:
        """Calculate overall safety score (0-1, higher is safer)"""
        
        # Safety score combines low false negative rate and high confidence
        fn_score = max(0, 1 - (false_negative_rate / 0.1))  # Penalty for >10% FN rate
        confidence_score = avg_confidence
        
        # Weighted combination (false negatives more critical)
        safety_score = (0.7 * fn_score) + (0.3 * confidence_score)
        
        return min(1.0, max(0.0, safety_score))
    
    def _get_age_group(self, age: Any) -> str:
        """Categorize patient age into groups"""
        
        if age == 'unknown' or age is None:
            return 'unknown'
        
        try:
            age_num = int(age)
            if age_num < 18:
                return 'pediatric'
            elif age_num < 65:
                return 'adult'
            else:
                return 'elderly'
        except (ValueError, TypeError):
            return 'unknown'
    
    def _calculate_bias_score(self, performance_by_group: Dict[str, float]) -> float:
        """Calculate demographic bias score (0-1, lower is better)"""
        
        if len(performance_by_group) < 2:
            return 0.0
        
        performances = list(performance_by_group.values())
        return np.std(performances)  # Standard deviation as bias metric
    
    async def _assess_medical_safety(self,
                                     medical_metrics: Dict[str, Any],
                                     stage_accuracies: Dict[str, float]) -> Dict[str, Any]:
        """Assess overall medical safety of the model"""
        
        safety_flags = []
        safety_score = medical_metrics['safety_score']
        
        # Check critical thresholds
        if medical_metrics['false_negative_rate'] > self.medical_thresholds['false_negative_threshold']:
            safety_flags.append("HIGH_FALSE_NEGATIVE_RATE")
        
        if medical_metrics['average_confidence'] < self.medical_thresholds['confidence_threshold']:
            safety_flags.append("LOW_AVERAGE_CONFIDENCE")
        
        if stage_accuracies.get('3', 0) < self.medical_thresholds['critical_stage_accuracy']:
            safety_flags.append("LOW_STAGE_3_ACCURACY")
        
        if stage_accuracies.get('4', 0) < self.medical_thresholds['critical_stage_accuracy']:
            safety_flags.append("LOW_STAGE_4_ACCURACY")
        
        # Overall safety assessment
        if safety_score >= 0.9:
            safety_level = "EXCELLENT"
        elif safety_score >= 0.8:
            safety_level = "GOOD"
        elif safety_score >= 0.7:
            safety_level = "ACCEPTABLE"
        elif safety_score >= 0.6:
            safety_level = "CONCERNING"
        else:
            safety_level = "UNSAFE"
        
        return {
            'safety_score': safety_score,
            'safety_level': safety_level,
            'safety_flags': safety_flags,
            'medical_approval_required': safety_level in ['CONCERNING', 'UNSAFE'] or len(safety_flags) > 0,
            'deployment_approved': safety_level in ['EXCELLENT', 'GOOD', 'ACCEPTABLE'] and len(safety_flags) == 0
        }
    
    async def _update_realtime_metrics(self,
                                       model_version: str,
                                       inference_data: Dict[str, Any]) -> None:
        """Update real-time performance metrics in Redis"""
        
        if not self.redis_client:
            return
        
        try:
            # Daily metrics
            today = datetime.now().strftime('%Y%m%d')
            metrics_key = f"vigia:metrics:daily:{model_version}:{today}"
            
            # Increment counters
            await self.redis_client.hincrby(metrics_key, "total_inferences", 1)
            
            if inference_data.get('correct_prediction'):
                await self.redis_client.hincrby(metrics_key, "correct_predictions", 1)
            
            # Update confidence running average
            if inference_data.get('max_confidence'):
                await self._update_running_average(
                    f"{metrics_key}:confidence",
                    inference_data['max_confidence']
                )
            
            # Set expiration (30 days)
            await self.redis_client.expire(metrics_key, 30 * 24 * 3600)
            
        except Exception as e:
            logger.error(f"Failed to update realtime metrics: {e}")
    
    async def _update_running_average(self, key: str, new_value: float) -> None:
        """Update running average in Redis"""
        
        pipe = self.redis_client.pipeline()
        current_avg = await self.redis_client.hget(key, "average")
        current_count = await self.redis_client.hget(key, "count")
        
        if current_avg and current_count:
            avg = float(current_avg)
            count = int(current_count)
            new_avg = ((avg * count) + new_value) / (count + 1)
            new_count = count + 1
        else:
            new_avg = new_value
            new_count = 1
        
        pipe.hset(key, "average", new_avg)
        pipe.hset(key, "count", new_count)
        await pipe.execute()
    
    async def _store_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Store comprehensive evaluation results"""
        
        # Store in training database
        try:
            # Find model version ID
            active_versions = await training_db.get_active_model_versions()
            version_id = None
            
            for version in active_versions:
                if version['version_name'] == results['model_version']:
                    version_id = version['id']
                    break
            
            if version_id:
                # Update model performance metrics
                metrics = {
                    'overall_accuracy': results['overall_accuracy'],
                    'precision_score': results['classification_report']['weighted avg']['precision'],
                    'recall_score': results['classification_report']['weighted avg']['recall'],
                    'f1_score': results['weighted_f1'],
                    **{f"accuracy_stage_{grade}": results['stage_accuracies'].get(grade, 0.0) 
                       for grade in ['0', '1', '2', '3', '4']},
                    'accuracy_unstageable': results['stage_accuracies'].get('U', 0.0),
                    'accuracy_dti': results['stage_accuracies'].get('DTI', 0.0)
                }
                
                await training_db.update_model_performance(version_id, metrics)
                
        except Exception as e:
            logger.error(f"Failed to store evaluation results: {e}")
    
    async def _check_performance_drift(self,
                                       model_version: str,
                                       current_results: Dict[str, Any]) -> None:
        """Check for model performance drift"""
        
        try:
            # Get historical performance
            if self.redis_client:
                history_key = f"vigia:performance_history:{model_version}"
                historical_data = await self.redis_client.lrange(history_key, 0, 4)  # Last 5 evaluations
                
                if len(historical_data) >= 2:
                    # Compare with recent performance
                    recent_accuracy = json.loads(historical_data[0])['overall_accuracy']
                    current_accuracy = current_results['overall_accuracy']
                    
                    accuracy_drop = recent_accuracy - current_accuracy
                    
                    if accuracy_drop > self.medical_thresholds['drift_threshold']:
                        # Performance drift detected
                        await self._alert_performance_drift(
                            model_version,
                            recent_accuracy,
                            current_accuracy,
                            accuracy_drop
                        )
                
                # Store current results in history
                await self.redis_client.lpush(
                    history_key,
                    json.dumps({
                        'timestamp': current_results['evaluation_timestamp'],
                        'overall_accuracy': current_results['overall_accuracy'],
                        'safety_score': current_results['medical_metrics']['safety_score']
                    }, default=str)
                )
                
                # Keep only last 10 evaluations
                await self.redis_client.ltrim(history_key, 0, 9)
                await self.redis_client.expire(history_key, 90 * 24 * 3600)  # 90 days
                
        except Exception as e:
            logger.error(f"Failed to check performance drift: {e}")
    
    async def _alert_performance_drift(self,
                                       model_version: str,
                                       previous_accuracy: float,
                                       current_accuracy: float,
                                       accuracy_drop: float) -> None:
        """Alert medical team about performance drift"""
        
        alert_data = {
            'alert_type': 'PERFORMANCE_DRIFT',
            'model_version': model_version,
            'previous_accuracy': previous_accuracy,
            'current_accuracy': current_accuracy,
            'accuracy_drop': accuracy_drop,
            'threshold': self.medical_thresholds['drift_threshold'],
            'timestamp': datetime.now().isoformat(),
            'requires_review': True
        }
        
        # Log critical alert
        logger.critical(f"PERFORMANCE DRIFT DETECTED: {model_version} - Accuracy dropped {accuracy_drop:.3f}")
        
        # Store alert in Redis for dashboard
        if self.redis_client:
            alert_key = f"vigia:alerts:drift:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await self.redis_client.setex(
                alert_key,
                7 * 24 * 3600,  # 7 days
                json.dumps(alert_data, default=str)
            )
        
        # Audit log
        await self.audit_service.log_activity(
            activity_type="performance_drift_alert",
            details=alert_data
        )
    
    async def get_model_dashboard_metrics(self, model_version: str) -> Dict[str, Any]:
        """Get real-time dashboard metrics for a model"""
        
        if not self.redis_client:
            # Mock data for development
            return {
                'daily_inferences': 245,
                'daily_accuracy': 0.92,
                'avg_confidence': 0.87,
                'safety_score': 0.89,
                'last_updated': datetime.now().isoformat()
            }
        
        today = datetime.now().strftime('%Y%m%d')
        metrics_key = f"vigia:metrics:daily:{model_version}:{today}"
        
        try:
            metrics = await self.redis_client.hgetall(metrics_key)
            
            total_inferences = int(metrics.get('total_inferences', 0))
            correct_predictions = int(metrics.get('correct_predictions', 0))
            
            daily_accuracy = correct_predictions / total_inferences if total_inferences > 0 else 0
            
            # Get confidence average
            confidence_key = f"{metrics_key}:confidence"
            confidence_data = await self.redis_client.hgetall(confidence_key)
            avg_confidence = float(confidence_data.get('average', 0))
            
            return {
                'daily_inferences': total_inferences,
                'daily_accuracy': daily_accuracy,
                'avg_confidence': avg_confidence,
                'correct_predictions': correct_predictions,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard metrics: {e}")
            return {}
    
    async def close(self) -> None:
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()


# Singleton instance
model_tracker = ModelPerformanceTracker()

__all__ = ['ModelPerformanceTracker', 'model_tracker']