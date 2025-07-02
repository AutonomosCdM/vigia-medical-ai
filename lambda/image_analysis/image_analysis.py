"""
Image Analysis Agent - Lambda Function
=====================================

AWS Lambda implementation of the Image Analysis Agent for medical image processing
using MONAI + YOLOv5 for pressure injury detection.
"""

import json
import logging
import boto3
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import base64
import io
from PIL import Image
import numpy as np

# AWS clients
dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
AGENTS_STATE_TABLE = os.environ['AGENTS_STATE_TABLE']
LPP_RESULTS_TABLE = os.environ['LPP_RESULTS_TABLE']
MEDICAL_STORAGE_BUCKET = os.environ['MEDICAL_STORAGE_BUCKET']

# DynamoDB tables
agents_table = dynamodb.Table(AGENTS_STATE_TABLE)
results_table = dynamodb.Table(LPP_RESULTS_TABLE)

class ImageAnalysisAgent:
    """Medical image analysis agent for pressure injury detection"""
    
    def __init__(self):
        self.agent_id = f"image_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
    async def analyze_medical_image_async(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze medical image for pressure injury detection
        
        Args:
            image_data: Image data with batman_token, image_path or image_bytes
            
        Returns:
            Analysis result with LPP classification and confidence
        """
        batman_token = image_data.get('batman_token')
        case_id = image_data.get('case_id')
        
        logger.info(f"🔬 Image Analysis Agent processing case: {case_id}")
        
        try:
            # Initialize analysis state
            analysis_state = {
                'agent_id': self.agent_id,
                'batman_token': batman_token,
                'case_id': case_id,
                'status': 'processing',
                'started_at': datetime.now().isoformat(),
                'analysis_type': 'lpp_detection'
            }
            
            await self._update_agent_state(analysis_state)
            
            # Load and validate image
            image_result = await self._load_and_validate_image(image_data)
            if not image_result['valid']:
                analysis_state['status'] = 'image_validation_failed'
                analysis_state['error'] = image_result['error']
                await self._update_agent_state(analysis_state)
                return analysis_state
            
            image = image_result['image']
            image_metadata = image_result['metadata']
            
            # Preprocess image for medical analysis
            preprocessed_image = await self._preprocess_medical_image(image)
            
            # Perform LPP detection (MONAI + YOLOv5 simulation)
            detection_result = await self._detect_pressure_injuries(preprocessed_image, image_metadata)
            
            # Validate results and determine confidence
            validation_result = await self._validate_detection_results(detection_result)
            
            # Store detection results
            await self._store_detection_results(batman_token, case_id, detection_result, validation_result)
            
            # Prepare final analysis result
            analysis_result = {
                'agent_id': self.agent_id,
                'batman_token': batman_token,
                'case_id': case_id,
                'status': 'completed',
                'analysis_type': 'lpp_detection',
                'detection_result': detection_result,
                'validation_result': validation_result,
                'confidence_score': validation_result['confidence'],
                'lpp_classification': detection_result['lpp_grade'],
                'requires_escalation': validation_result['requires_escalation'],
                'completed_at': datetime.now().isoformat(),
                'processing_time_ms': self._calculate_processing_time(analysis_state['started_at'])
            }
            
            await self._update_agent_state(analysis_result)
            
            logger.info(f"✅ Image Analysis completed for case: {case_id} - Grade: {detection_result['lpp_grade']}")
            return analysis_result
            
        except Exception as e:
            error_result = {
                'agent_id': self.agent_id,
                'batman_token': batman_token,
                'case_id': case_id,
                'status': 'error',
                'error': str(e),
                'error_timestamp': datetime.now().isoformat()
            }
            
            await self._update_agent_state(error_result)
            logger.error(f"❌ Image Analysis error for case {case_id}: {str(e)}")
            return error_result
    
    async def _load_and_validate_image(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load and validate medical image from various sources"""
        
        try:
            image = None
            metadata = {}
            
            # Load from S3 path
            if 'image_path' in image_data:
                image_path = image_data['image_path']
                if image_path.startswith('s3://'):
                    # Parse S3 path
                    bucket_key = image_path.replace('s3://', '').split('/', 1)
                    bucket = bucket_key[0]
                    key = bucket_key[1] if len(bucket_key) > 1 else ''
                    
                    response = s3.get_object(Bucket=bucket, Key=key)
                    image_bytes = response['Body'].read()
                    image = Image.open(io.BytesIO(image_bytes))
                    metadata['s3_bucket'] = bucket
                    metadata['s3_key'] = key
                else:
                    # Local file path (for development)
                    image = Image.open(image_path)
                    metadata['file_path'] = image_path
            
            # Load from base64 encoded data
            elif 'image_bytes' in image_data:
                image_bytes = base64.b64decode(image_data['image_bytes'])
                image = Image.open(io.BytesIO(image_bytes))
                metadata['source'] = 'base64_encoded'
            
            # Load from direct bytes
            elif 'image_data' in image_data:
                image_bytes = image_data['image_data']
                image = Image.open(io.BytesIO(image_bytes))
                metadata['source'] = 'direct_bytes'
            
            else:
                return {
                    'valid': False,
                    'error': 'No valid image source found in image_data'
                }
            
            # Validate image properties
            if image is None:
                return {
                    'valid': False,
                    'error': 'Failed to load image from provided source'
                }
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Validate image dimensions
            width, height = image.size
            if width < 100 or height < 100:
                return {
                    'valid': False,
                    'error': f'Image too small for analysis: {width}x{height}'
                }
            
            metadata.update({
                'width': width,
                'height': height,
                'mode': image.mode,
                'format': getattr(image, 'format', 'Unknown')
            })
            
            return {
                'valid': True,
                'image': image,
                'metadata': metadata
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Image validation error: {str(e)}'
            }
    
    async def _preprocess_medical_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for medical analysis"""
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Resize to standard medical analysis size
        from PIL import Image as PILImage
        image_resized = image.resize((416, 416), PILImage.Resampling.LANCZOS)
        image_array = np.array(image_resized)
        
        # Normalize pixel values
        image_array = image_array.astype(np.float32) / 255.0
        
        # Apply medical-specific preprocessing
        # Note: In production, this would include CLAHE enhancement, 
        # noise reduction, and other medical imaging techniques
        
        return image_array
    
    async def _detect_pressure_injuries(self, image: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect pressure injuries using MONAI + YOLOv5 (simulated for Lambda environment)
        """
        
        # Simulate medical AI detection (in production, this would use actual MONAI/YOLOv5)
        import random
        
        # Simulate detection based on image characteristics
        image_mean = np.mean(image)
        image_std = np.std(image)
        
        # Generate realistic LPP detection results
        confidence_base = 0.75 + (image_std * 0.3)  # Higher variance often indicates pathology
        confidence = min(confidence_base + random.uniform(-0.15, 0.15), 0.99)
        
        # Determine LPP grade based on simulated analysis
        if image_mean < 0.3:  # Darker images might indicate more severe injuries
            lpp_grade = random.choice([2, 3, 4])
        elif image_mean > 0.7:  # Lighter images might be early stage or no injury
            lpp_grade = random.choice([0, 1])
        else:
            lpp_grade = random.choice([1, 2])
        
        # Generate detection details
        detection_result = {
            'lpp_grade': lpp_grade,
            'confidence': confidence,
            'detection_method': 'MONAI_YOLOv5_simulation',
            'bounding_boxes': [
                {
                    'x': random.randint(50, 200),
                    'y': random.randint(50, 200),
                    'width': random.randint(80, 150),
                    'height': random.randint(80, 150),
                    'confidence': confidence
                }
            ] if lpp_grade > 0 else [],
            'anatomical_location': random.choice(['sacrum', 'heel', 'shoulder', 'hip', 'elbow']),
            'tissue_assessment': {
                'erythema_present': lpp_grade > 0,
                'tissue_loss': lpp_grade > 1,
                'depth_assessment': ['superficial', 'partial_thickness', 'full_thickness', 'deep_tissue'][min(lpp_grade, 3)]
            },
            'image_quality_metrics': {
                'resolution': f"{metadata.get('width', 0)}x{metadata.get('height', 0)}",
                'brightness': float(image_mean),
                'contrast': float(image_std),
                'quality_score': confidence
            }
        }
        
        return detection_result
    
    async def _validate_detection_results(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate detection results and determine escalation requirements"""
        
        confidence = detection_result['confidence']
        lpp_grade = detection_result['lpp_grade']
        
        # Determine validation status
        confidence_threshold = 0.6
        high_confidence = confidence >= confidence_threshold
        
        # Check escalation criteria
        escalation_criteria = [
            confidence < confidence_threshold,  # Low confidence
            lpp_grade >= 3,  # High-grade LPP
            len(detection_result.get('bounding_boxes', [])) > 3,  # Multiple lesions
        ]
        
        requires_escalation = any(escalation_criteria)
        
        # Clinical validation
        clinical_indicators = {
            'confidence_adequate': high_confidence,
            'grade_severe': lpp_grade >= 3,
            'multiple_lesions': len(detection_result.get('bounding_boxes', [])) > 1,
            'anatomical_high_risk': detection_result.get('anatomical_location') in ['sacrum', 'heel']
        }
        
        validation_result = {
            'confidence': confidence,
            'confidence_adequate': high_confidence,
            'requires_escalation': requires_escalation,
            'escalation_reasons': [reason for reason, meets in zip(
                ['low_confidence', 'high_grade_lpp', 'multiple_lesions'],
                escalation_criteria
            ) if meets],
            'clinical_indicators': clinical_indicators,
            'recommended_action': self._get_recommended_action(lpp_grade, confidence),
            'validation_timestamp': datetime.now().isoformat()
        }
        
        return validation_result
    
    def _get_recommended_action(self, lpp_grade: int, confidence: float) -> str:
        """Get recommended clinical action based on detection results"""
        
        if lpp_grade == 0:
            return "Continue routine monitoring and prevention protocols"
        elif lpp_grade == 1:
            return "Enhanced monitoring, pressure relief, and skin protection"
        elif lpp_grade == 2:
            return "Wound care protocols and pressure redistribution"
        elif lpp_grade == 3:
            return "Advanced wound care and medical team review"
        elif lpp_grade == 4:
            return "EMERGENCY - Immediate surgical consultation required"
        else:
            return "Clinical assessment required"
    
    async def _store_detection_results(self, batman_token: str, case_id: str, 
                                     detection_result: Dict[str, Any], 
                                     validation_result: Dict[str, Any]) -> None:
        """Store detection results in DynamoDB"""
        
        result_item = {
            'case_id': case_id,
            'analysis_timestamp': int(datetime.now().timestamp() * 1000),
            'batman_token': batman_token,
            'agent_id': self.agent_id,
            'detection_result': json.dumps(detection_result),
            'validation_result': json.dumps(validation_result),
            'lpp_grade': detection_result['lpp_grade'],
            'confidence': validation_result['confidence'],
            'requires_escalation': validation_result['requires_escalation'],
            'created_at': datetime.now().isoformat()
        }
        
        results_table.put_item(Item=result_item)
    
    async def _update_agent_state(self, state: Dict[str, Any]) -> None:
        """Update agent state in DynamoDB"""
        
        item = {
            'batman_token': state['batman_token'],
            'agent_timestamp': int(datetime.now().timestamp() * 1000),
            'agent_type': 'image_analysis',
            'agent_id': self.agent_id,
            'state_data': json.dumps(state),
            'case_id': state.get('case_id'),
            'status': state.get('status'),
            'updated_at': datetime.now().isoformat()
        }
        
        agents_table.put_item(Item=item)
    
    def _calculate_processing_time(self, start_time: str) -> int:
        """Calculate processing time in milliseconds"""
        start = datetime.fromisoformat(start_time)
        end = datetime.now()
        return int((end - start).total_seconds() * 1000)

# Global agent instance
image_agent = ImageAnalysisAgent()

def handler(event, context):
    """
    AWS Lambda handler for Image Analysis Agent
    
    Args:
        event: Lambda event containing image_data
        context: Lambda context
        
    Returns:
        Image analysis result
    """
    logger.info(f"🔬 Image Analysis Agent Lambda triggered")
    logger.info(f"Event: {json.dumps(event, default=str)}")
    
    try:
        # Extract image data from event
        if 'image_data' in event:
            image_data = event['image_data']
        else:
            image_data = event
        
        # Ensure required fields
        if 'case_id' not in image_data:
            image_data['case_id'] = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if 'batman_token' not in image_data:
            image_data['batman_token'] = f"BATMAN-IMG-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Process image analysis synchronously
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                image_agent.analyze_medical_image_async(image_data)
            )
        finally:
            loop.close()
        
        logger.info(f"✅ Image analysis completed successfully")
        
        return {
            'statusCode': 200,
            'body': result
        }
        
    except Exception as e:
        logger.error(f"❌ Image Analysis Agent Lambda error: {str(e)}")
        
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'agent_id': image_agent.agent_id,
                'timestamp': datetime.now().isoformat()
            }
        }