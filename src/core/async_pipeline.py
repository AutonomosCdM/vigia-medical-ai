"""
Asynchronous Medical Pipeline Orchestrator
==========================================

Orquestador principal del pipeline médico asíncrono que coordina tareas
Celery para prevenir timeouts y garantizar procesamiento fluido.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from src.core.celery_config import celery_app
from src.utils.secure_logger import SecureLogger
from src.utils.failure_handler import log_task_failure

# PHI Tokenization Integration
from src.core.phi_tokenization_client import tokenize_patient_phi, TokenizedPatient

logger = SecureLogger(__name__)

class AsyncMedicalPipeline:
    """
    Orquestador del pipeline médico asíncrono
    """
    
    def __init__(self):
        self.logger = SecureLogger(__name__)
        
    async def process_medical_case_async(
        self,
        image_path: str,
        hospital_mrn: str,
        patient_context: Optional[Dict[str, Any]] = None,
        processing_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Procesa caso médico de forma completamente asíncrona con tokenización PHI
        
        Args:
            image_path: Ruta de la imagen médica
            hospital_mrn: Hospital MRN (Bruce Wayne data - will be tokenized to Batman)
            patient_context: Contexto médico del paciente
            processing_options: Opciones de procesamiento
            
        Returns:
            Dict con IDs de tareas asíncronas y estado inicial
        """
        try:
            self.logger.info(f"Starting async medical pipeline with PHI tokenization for MRN: {hospital_mrn[:8]}...")
            
            # FASE 1: PHI TOKENIZATION - Convert Bruce Wayne → Batman
            try:
                tokenized_patient = await tokenize_patient_phi(
                    hospital_mrn=hospital_mrn,
                    request_purpose="LPP detection and medical analysis"
                )
                
                # Use Batman token for all processing
                token_id = tokenized_patient.token_id
                patient_alias = tokenized_patient.patient_alias
                
                self.logger.audit("phi_tokenization_successful", {
                    "hospital_mrn": hospital_mrn[:8] + "...",
                    "token_id": token_id,
                    "patient_alias": patient_alias,
                    "tokenization_purpose": "async_medical_pipeline"
                })
                
            except Exception as tokenization_error:
                self.logger.error(f"PHI tokenization failed: {tokenization_error}")
                return {
                    'success': False,
                    'error': 'PHI tokenization failed - cannot process without proper data protection',
                    'hospital_mrn': hospital_mrn[:8] + "...",
                    'tokenization_error': str(tokenization_error)
                }
            
            self.logger.info(f"Processing medical case for tokenized patient: {patient_alias} (Token: {token_id})")
            
            # Importar tareas asíncronas
            from src.tasks.medical import (
                image_analysis_task, 
                risk_score_task, 
                medical_analysis_task,
                triage_task
            )
            from src.tasks.audit import audit_log_task
            from src.tasks.notifications import notify_slack_task
            
            # Configuración por defecto
            options = processing_options or {}
            analysis_type = options.get('analysis_type', 'complete')
            notify_channels = options.get('notify_channels', [])
            
            # 1. Iniciar análisis de imagen (tarea principal) - WITH BATMAN TOKEN
            image_task = image_analysis_task.delay(
                image_path=image_path,
                token_id=token_id,  # Batman token instead of PHI
                patient_context=patient_context,
                tokenized_patient_data=tokenized_patient.to_dict()
            )
            
            # 2. Análisis médico completo (depende de análisis de imagen) - WITH BATMAN TOKEN  
            medical_task = medical_analysis_task.delay(
                image_path=image_path,
                token_id=token_id,  # Batman token instead of PHI
                analysis_type=analysis_type,
                patient_context=patient_context,
                tokenized_patient_data=tokenized_patient.to_dict()
            )
            
            # 3. Logging de auditoría (inmediato) - WITH BATMAN TOKEN
            audit_task = audit_log_task.delay(
                event_data={
                    'pipeline_started': True,
                    'token_id': token_id,  # Batman token for audit
                    'patient_alias': patient_alias,
                    'image_path': image_path,
                    'analysis_type': analysis_type,
                    'phi_tokenization_completed': True
                },
                event_type='medical_pipeline_start',
                patient_context=patient_context
            )
            
            # Pipeline result tracking - WITH BATMAN TOKEN
            pipeline_result = {
                'success': True,
                'pipeline_id': f"pipeline_{token_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                'token_id': token_id,  # Batman token instead of PHI
                'patient_alias': patient_alias,
                'started_at': datetime.now(timezone.utc).isoformat(),
                'status': 'processing',
                'task_ids': {
                    'image_analysis': image_task.id,
                    'medical_analysis': medical_task.id,
                    'audit_log': audit_task.id
                },
                'processing_options': options,
                'phi_tokenization': {
                    'hospital_mrn_partial': hospital_mrn[:8] + "...",
                    'tokenized_as': patient_alias,
                    'expires_at': tokenized_patient.expires_at.isoformat()
                }
            }
            
            self.logger.info(f"Async pipeline initiated for tokenized patient: {patient_alias} (Token: {token_id}) - Pipeline ID: {pipeline_result['pipeline_id']}")
            return pipeline_result
            
        except Exception as e:
            # Log del fallo del pipeline
            failure_response = log_task_failure(
                task_name='async_medical_pipeline',
                task_id='pipeline_orchestrator',
                exception=e,
                context={
                    'image_path': image_path,
                    'patient_code': patient_code,
                    'processing_options': processing_options
                },
                patient_context=patient_context
            )
            
            return {
                'success': False,
                'error': str(e),
                'hospital_mrn_partial': hospital_mrn[:8] + "..." if 'hospital_mrn' in locals() else 'unknown',
                'failure_logged': True,
                'phi_tokenization_attempted': True
            }
    
    def get_pipeline_status(self, pipeline_id: str, task_ids: Dict[str, str]) -> Dict[str, Any]:
        """
        Obtiene estado actual de todas las tareas del pipeline
        
        Args:
            pipeline_id: ID del pipeline
            task_ids: Dict con IDs de las tareas
            
        Returns:
            Dict con estado completo del pipeline
        """
        try:
            status_result = {
                'pipeline_id': pipeline_id,
                'overall_status': 'processing',
                'checked_at': datetime.now(timezone.utc).isoformat(),
                'tasks_status': {},
                'completed_tasks': 0,
                'total_tasks': len(task_ids),
                'has_failures': False
            }
            
            # Verificar estado de cada tarea
            for task_name, task_id in task_ids.items():
                try:
                    task_result = celery_app.AsyncResult(task_id)
                    task_status = {
                        'task_id': task_id,
                        'status': task_result.status,
                        'ready': task_result.ready(),
                        'successful': task_result.successful() if task_result.ready() else None,
                        'failed': task_result.failed() if task_result.ready() else None
                    }
                    
                    # Agregar resultado si está disponible
                    if task_result.ready() and task_result.successful():
                        task_status['result'] = task_result.result
                        status_result['completed_tasks'] += 1
                    elif task_result.failed():
                        task_status['error'] = str(task_result.info)
                        status_result['has_failures'] = True
                    
                    status_result['tasks_status'][task_name] = task_status
                    
                except Exception as task_error:
                    status_result['tasks_status'][task_name] = {
                        'task_id': task_id,
                        'status': 'ERROR',
                        'error': str(task_error)
                    }
                    status_result['has_failures'] = True
            
            # Determinar estado general
            if status_result['has_failures']:
                status_result['overall_status'] = 'failed'
            elif status_result['completed_tasks'] == status_result['total_tasks']:
                status_result['overall_status'] = 'completed'
            else:
                status_result['overall_status'] = 'processing'
            
            return status_result
            
        except Exception as e:
            self.logger.error(f"Failed to get pipeline status: {e}")
            return {
                'pipeline_id': pipeline_id,
                'overall_status': 'error',
                'error': str(e),
                'checked_at': datetime.now(timezone.utc).isoformat()
            }
    
    def wait_for_pipeline_completion(
        self,
        pipeline_id: str,
        task_ids: Dict[str, str],
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Espera completación del pipeline con timeout
        
        Args:
            pipeline_id: ID del pipeline
            task_ids: Dict con IDs de las tareas
            timeout: Timeout en segundos
            
        Returns:
            Dict con resultados finales del pipeline
        """
        try:
            self.logger.info(f"Waiting for pipeline completion: {pipeline_id}")
            
            # Crear grupo de tareas para espera sincronizada
            from celery import group
            task_group = group([
                celery_app.AsyncResult(task_id) 
                for task_id in task_ids.values()
            ])
            
            # Esperar con timeout
            results = task_group.get(timeout=timeout, propagate=False)
            
            # Procesar resultados
            final_results = {
                'pipeline_id': pipeline_id,
                'completed_at': datetime.now(timezone.utc).isoformat(),
                'success': True,
                'timeout': False,
                'results': {}
            }
            
            # Mapear resultados por nombre de tarea
            for i, (task_name, task_id) in enumerate(task_ids.items()):
                if i < len(results):
                    final_results['results'][task_name] = results[i]
                else:
                    final_results['results'][task_name] = {'error': 'No result available'}
                    final_results['success'] = False
            
            self.logger.info(f"Pipeline completed: {pipeline_id}")
            return final_results
            
        except Exception as e:
            if 'timeout' in str(e).lower():
                self.logger.warning(f"Pipeline timeout: {pipeline_id}")
                return {
                    'pipeline_id': pipeline_id,
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                    'success': False,
                    'timeout': True,
                    'error': 'Pipeline timeout'
                }
            else:
                self.logger.error(f"Pipeline completion error: {e}")
                return {
                    'pipeline_id': pipeline_id,
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                    'success': False,
                    'timeout': False,
                    'error': str(e)
                }
    
    def trigger_escalation_pipeline(
        self,
        escalation_data: Dict[str, Any],
        escalation_type: str,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Dispara pipeline de escalación médica
        
        Args:
            escalation_data: Datos de la escalación
            escalation_type: Tipo de escalación
            patient_context: Contexto del paciente
            
        Returns:
            Dict con IDs de tareas de escalación
        """
        try:
            self.logger.info(f"Triggering escalation pipeline: {escalation_type}")
            
            # Importar tareas de escalación
            from src.tasks.notifications import (
                medical_alert_slack_task,
                escalation_notification_task
            )
            from src.tasks.audit import medical_decision_audit_task
            
            # Determinar canales según tipo de escalación
            target_channels = self._get_escalation_channels(escalation_type)
            target_roles = self._get_escalation_roles(escalation_type)
            
            # Disparar tareas de escalación - WITH BATMAN TOKEN
            alert_task = medical_alert_slack_task.delay(
                alert_data=escalation_data,
                alert_type=escalation_type,
                token_id=patient_context.get('token_id', 'unknown') if patient_context else 'unknown',
                patient_alias=patient_context.get('patient_alias', 'unknown') if patient_context else 'unknown',
                medical_team_channels=target_channels
            )
            
            escalation_task = escalation_notification_task.delay(
                escalation_data=escalation_data,
                escalation_type=escalation_type,
                target_roles=target_roles,
                token_id=patient_context.get('token_id', 'unknown') if patient_context else 'unknown'
            )
            
            # Auditoría de escalación - WITH BATMAN TOKEN
            audit_task = medical_decision_audit_task.delay(
                decision_data=escalation_data,
                token_id=patient_context.get('token_id', 'unknown') if patient_context else 'unknown',
                patient_alias=patient_context.get('patient_alias', 'unknown') if patient_context else 'unknown',
                medical_context={'escalation_triggered': True, 'escalation_type': escalation_type}
            )
            
            escalation_result = {
                'success': True,
                'escalation_id': f"escalation_{escalation_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                'escalation_type': escalation_type,
                'started_at': datetime.now(timezone.utc).isoformat(),
                'task_ids': {
                    'medical_alert': alert_task.id,
                    'escalation_notification': escalation_task.id,
                    'audit_log': audit_task.id
                },
                'target_channels': target_channels,
                'target_roles': target_roles
            }
            
            self.logger.info(f"Escalation pipeline initiated: {escalation_result['escalation_id']}")
            return escalation_result
            
        except Exception as e:
            self.logger.error(f"Failed to trigger escalation pipeline: {e}")
            return {
                'success': False,
                'escalation_type': escalation_type,
                'error': str(e)
            }
    
    def _get_escalation_channels(self, escalation_type: str) -> List[str]:
        """Obtiene canales según tipo de escalación"""
        escalation_channels = {
            'human_review': ['#equipo-medico'],
            'specialist_review': ['#especialistas', '#equipo-medico'],
            'emergency': ['#emergencias', '#especialistas', '#equipo-medico'],
            'system_error': ['#sistema-alertas'],
            'high_risk': ['#riesgo-alto', '#equipo-medico']
        }
        return escalation_channels.get(escalation_type, ['#general-medico'])
    
    def _get_escalation_roles(self, escalation_type: str) -> List[str]:
        """Obtiene roles según tipo de escalación"""
        escalation_roles = {
            'human_review': ['medical_team'],
            'specialist_review': ['specialists', 'medical_team'],
            'emergency': ['emergency', 'specialists', 'medical_team'],
            'system_error': ['admin'],
            'high_risk': ['medical_team', 'nursing_team']
        }
        return escalation_roles.get(escalation_type, ['medical_team'])

# Instancia global del pipeline
async_pipeline = AsyncMedicalPipeline()