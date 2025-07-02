"""
VIGIA Medical AI - Training Database
====================================

Production-grade database infrastructure for medical model training.
Supports HIPAA-compliant storage, NPUAP grading, and training pipeline integration.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import hashlib
import asyncpg
import aioredis
from enum import Enum
import numpy as np
from PIL import Image
import os

logger = logging.getLogger(__name__)

class NPUAPGrade(Enum):
    """NPUAP Pressure Injury Classification"""
    STAGE_0 = "0"  # No visible pressure injury
    STAGE_1 = "1"  # Non-blanchable erythema
    STAGE_2 = "2"  # Partial thickness skin loss
    STAGE_3 = "3"  # Full thickness skin loss
    STAGE_4 = "4"  # Full thickness skin and tissue loss
    UNSTAGEABLE = "U"  # Unstageable/unclassified
    DEEP_TISSUE = "DTI"  # Deep tissue pressure injury

class TrainingDatabase:
    """Medical training database with HIPAA compliance"""
    
    def __init__(self, 
                 postgres_url: str = None,
                 redis_url: str = None,
                 storage_path: str = "./data/training"):
        self.postgres_url = postgres_url or os.getenv("POSTGRES_URL", "postgresql://localhost/vigia_training")
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/2")
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Connection pools
        self.pg_pool = None
        self.redis_client = None
        self._initialized = False
        
        logger.info(f"TrainingDatabase initialized - Storage: {self.storage_path}")
    
    async def initialize(self) -> None:
        """Initialize database connections and schema"""
        try:
            # PostgreSQL connection pool
            self.pg_pool = await asyncpg.create_pool(
                self.postgres_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Redis connection
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                decode_responses=True
            )
            
            # Create database schema
            await self._create_schema()
            
            self._initialized = True
            logger.info("TrainingDatabase initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TrainingDatabase: {e}")
            # Fallback to mock mode for development
            await self._initialize_mock_mode()
    
    async def _initialize_mock_mode(self) -> None:
        """Initialize in mock mode for development"""
        self.pg_pool = None
        self.redis_client = None
        self._initialized = True
        self._mock_mode = True
        logger.warning("TrainingDatabase initialized in MOCK MODE")
    
    async def _create_schema(self) -> None:
        """Create database schema for medical training data"""
        schema_sql = """
        -- Medical Images Table
        CREATE TABLE IF NOT EXISTS medical_images (
            id SERIAL PRIMARY KEY,
            image_hash VARCHAR(64) UNIQUE NOT NULL,
            batman_token VARCHAR(100),
            file_path TEXT NOT NULL,
            npuap_grade VARCHAR(10) NOT NULL,
            body_location VARCHAR(50),
            patient_age INTEGER,
            patient_gender VARCHAR(10),
            image_quality_score FLOAT,
            annotated_by VARCHAR(100),
            annotation_confidence FLOAT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            metadata JSONB
        );
        
        -- Model Versions Table
        CREATE TABLE IF NOT EXISTS model_versions (
            id SERIAL PRIMARY KEY,
            version_name VARCHAR(100) UNIQUE NOT NULL,
            model_type VARCHAR(50) NOT NULL, -- 'MONAI', 'YOLOv5', 'MedGemma'
            model_path TEXT NOT NULL,
            training_dataset_size INTEGER,
            accuracy_stage_0 FLOAT,
            accuracy_stage_1 FLOAT,
            accuracy_stage_2 FLOAT,
            accuracy_stage_3 FLOAT,
            accuracy_stage_4 FLOAT,
            accuracy_unstageable FLOAT,
            accuracy_dti FLOAT,
            overall_accuracy FLOAT,
            precision_score FLOAT,
            recall_score FLOAT,
            f1_score FLOAT,
            training_started_at TIMESTAMP,
            training_completed_at TIMESTAMP,
            deployed_at TIMESTAMP,
            is_active BOOLEAN DEFAULT FALSE,
            performance_notes TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        -- Training Sessions Table
        CREATE TABLE IF NOT EXISTS training_sessions (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(100) UNIQUE NOT NULL,
            model_version_id INTEGER REFERENCES model_versions(id),
            dataset_version VARCHAR(50),
            training_config JSONB,
            started_at TIMESTAMP DEFAULT NOW(),
            completed_at TIMESTAMP,
            status VARCHAR(20) DEFAULT 'running', -- 'running', 'completed', 'failed'
            final_metrics JSONB,
            error_message TEXT,
            total_epochs INTEGER,
            best_epoch INTEGER,
            early_stopped BOOLEAN DEFAULT FALSE,
            gpu_hours_used FLOAT,
            created_by VARCHAR(100)
        );
        
        -- Model Performance Tracking
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            model_version_id INTEGER REFERENCES model_versions(id),
            evaluation_date DATE DEFAULT CURRENT_DATE,
            test_dataset_size INTEGER,
            confusion_matrix JSONB,
            classification_report JSONB,
            roi_analysis JSONB,
            clinical_validation_notes TEXT,
            validated_by VARCHAR(100),
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        -- Data Lineage and Audit
        CREATE TABLE IF NOT EXISTS data_audit (
            id SERIAL PRIMARY KEY,
            operation_type VARCHAR(50) NOT NULL, -- 'insert', 'update', 'delete', 'export'
            table_name VARCHAR(50) NOT NULL,
            record_id INTEGER,
            batman_token VARCHAR(100),
            operation_metadata JSONB,
            performed_by VARCHAR(100) NOT NULL,
            performed_at TIMESTAMP DEFAULT NOW(),
            compliance_notes TEXT
        );
        
        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_medical_images_batman_token ON medical_images(batman_token);
        CREATE INDEX IF NOT EXISTS idx_medical_images_npuap_grade ON medical_images(npuap_grade);
        CREATE INDEX IF NOT EXISTS idx_medical_images_created_at ON medical_images(created_at);
        CREATE INDEX IF NOT EXISTS idx_model_versions_active ON model_versions(is_active);
        CREATE INDEX IF NOT EXISTS idx_training_sessions_status ON training_sessions(status);
        CREATE INDEX IF NOT EXISTS idx_data_audit_batman_token ON data_audit(batman_token);
        """
        
        if self.pg_pool:
            async with self.pg_pool.acquire() as conn:
                await conn.execute(schema_sql)
            logger.info("Database schema created successfully")
    
    async def store_medical_image(self,
                                  image_path: str,
                                  npuap_grade: NPUAPGrade,
                                  batman_token: str = None,
                                  body_location: str = None,
                                  patient_age: int = None,
                                  patient_gender: str = None,
                                  annotated_by: str = "system",
                                  annotation_confidence: float = 1.0,
                                  metadata: Dict[str, Any] = None) -> Optional[int]:
        """Store medical image with NPUAP classification"""
        
        try:
            # Calculate image hash for deduplication
            image_hash = await self._calculate_image_hash(image_path)
            
            # Store image file securely
            secure_path = await self._store_image_file(image_path, image_hash)
            
            # Assess image quality
            quality_score = await self._assess_image_quality(image_path)
            
            if hasattr(self, '_mock_mode') and self._mock_mode:
                # Mock mode for development
                record_id = hash(image_hash) % 10000
                logger.info(f"Mock stored medical image: {secure_path} (Grade: {npuap_grade.value})")
                return record_id
            
            # Store in database
            insert_sql = """
            INSERT INTO medical_images 
            (image_hash, batman_token, file_path, npuap_grade, body_location, 
             patient_age, patient_gender, image_quality_score, annotated_by, 
             annotation_confidence, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING id
            """
            
            async with self.pg_pool.acquire() as conn:
                record_id = await conn.fetchval(
                    insert_sql,
                    image_hash, batman_token, str(secure_path), npuap_grade.value,
                    body_location, patient_age, patient_gender, quality_score,
                    annotated_by, annotation_confidence, json.dumps(metadata or {})
                )
            
            # Audit log
            await self._log_audit("insert", "medical_images", record_id, batman_token)
            
            # Cache training statistics
            await self._update_training_stats_cache()
            
            logger.info(f"Stored medical image: {secure_path} (ID: {record_id}, Grade: {npuap_grade.value})")
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to store medical image: {e}")
            return None
    
    async def get_training_dataset(self,
                                   npuap_grades: List[NPUAPGrade] = None,
                                   min_quality_score: float = 0.7,
                                   limit: int = None) -> List[Dict[str, Any]]:
        """Get training dataset with filtering"""
        
        if hasattr(self, '_mock_mode') and self._mock_mode:
            # Mock training data
            return [
                {
                    'id': 1,
                    'file_path': 'mock/image1.jpg',
                    'npuap_grade': NPUAPGrade.STAGE_2.value,
                    'image_quality_score': 0.95,
                    'metadata': {'mock': True}
                },
                {
                    'id': 2,
                    'file_path': 'mock/image2.jpg',
                    'npuap_grade': NPUAPGrade.STAGE_1.value,
                    'image_quality_score': 0.87,
                    'metadata': {'mock': True}
                }
            ]
        
        # Build query with filters
        where_conditions = ["image_quality_score >= $1"]
        params = [min_quality_score]
        param_count = 1
        
        if npuap_grades:
            param_count += 1
            grade_values = [grade.value for grade in npuap_grades]
            where_conditions.append(f"npuap_grade = ANY(${param_count})")
            params.append(grade_values)
        
        limit_clause = f" LIMIT ${param_count + 1}" if limit else ""
        if limit:
            params.append(limit)
        
        query = f"""
        SELECT id, file_path, npuap_grade, body_location, patient_age, 
               patient_gender, image_quality_score, metadata, created_at
        FROM medical_images 
        WHERE {' AND '.join(where_conditions)}
        ORDER BY image_quality_score DESC, created_at DESC
        {limit_clause}
        """
        
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    async def create_model_version(self,
                                   version_name: str,
                                   model_type: str,
                                   model_path: str,
                                   training_config: Dict[str, Any] = None) -> Optional[int]:
        """Create new model version record"""
        
        if hasattr(self, '_mock_mode') and self._mock_mode:
            mock_id = hash(version_name) % 1000
            logger.info(f"Mock created model version: {version_name} (ID: {mock_id})")
            return mock_id
        
        insert_sql = """
        INSERT INTO model_versions (version_name, model_type, model_path, training_started_at)
        VALUES ($1, $2, $3, NOW())
        RETURNING id
        """
        
        try:
            async with self.pg_pool.acquire() as conn:
                version_id = await conn.fetchval(insert_sql, version_name, model_type, model_path)
            
            # Create training session
            session_id = f"{version_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await self._create_training_session(version_id, session_id, training_config)
            
            logger.info(f"Created model version: {version_name} (ID: {version_id})")
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to create model version: {e}")
            return None
    
    async def update_model_performance(self,
                                       version_id: int,
                                       metrics: Dict[str, float]) -> bool:
        """Update model performance metrics"""
        
        if hasattr(self, '_mock_mode') and self._mock_mode:
            logger.info(f"Mock updated model performance for version {version_id}")
            return True
        
        update_sql = """
        UPDATE model_versions SET
            accuracy_stage_0 = $2,
            accuracy_stage_1 = $3,
            accuracy_stage_2 = $4,
            accuracy_stage_3 = $5,
            accuracy_stage_4 = $6,
            accuracy_unstageable = $7,
            accuracy_dti = $8,
            overall_accuracy = $9,
            precision_score = $10,
            recall_score = $11,
            f1_score = $12,
            updated_at = NOW()
        WHERE id = $1
        """
        
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute(
                    update_sql,
                    version_id,
                    metrics.get('accuracy_stage_0', 0.0),
                    metrics.get('accuracy_stage_1', 0.0),
                    metrics.get('accuracy_stage_2', 0.0),
                    metrics.get('accuracy_stage_3', 0.0),
                    metrics.get('accuracy_stage_4', 0.0),
                    metrics.get('accuracy_unstageable', 0.0),
                    metrics.get('accuracy_dti', 0.0),
                    metrics.get('overall_accuracy', 0.0),
                    metrics.get('precision_score', 0.0),
                    metrics.get('recall_score', 0.0),
                    metrics.get('f1_score', 0.0)
                )
            
            logger.info(f"Updated performance metrics for model version {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model performance: {e}")
            return False
    
    async def get_active_model_versions(self) -> List[Dict[str, Any]]:
        """Get currently active model versions"""
        
        if hasattr(self, '_mock_mode') and self._mock_mode:
            return [
                {
                    'id': 1,
                    'version_name': 'MONAI_v1.2.3',
                    'model_type': 'MONAI',
                    'overall_accuracy': 0.962,
                    'deployed_at': datetime.now() - timedelta(days=5)
                }
            ]
        
        query = """
        SELECT id, version_name, model_type, model_path, overall_accuracy,
               precision_score, recall_score, f1_score, deployed_at, created_at
        FROM model_versions 
        WHERE is_active = true
        ORDER BY deployed_at DESC
        """
        
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(query)
            return [dict(row) for row in rows]
    
    async def _calculate_image_hash(self, image_path: str) -> str:
        """Calculate SHA-256 hash of image file"""
        hash_sha256 = hashlib.sha256()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    async def _store_image_file(self, source_path: str, image_hash: str) -> Path:
        """Store image file securely with hash-based naming"""
        # Create subdirectory based on first 2 chars of hash
        subdir = self.storage_path / image_hash[:2]
        subdir.mkdir(exist_ok=True)
        
        # Secure filename with hash
        file_ext = Path(source_path).suffix
        secure_path = subdir / f"{image_hash}{file_ext}"
        
        # Copy file if not already exists
        if not secure_path.exists():
            import shutil
            shutil.copy2(source_path, secure_path)
        
        return secure_path
    
    async def _assess_image_quality(self, image_path: str) -> float:
        """Assess image quality using basic metrics"""
        try:
            with Image.open(image_path) as img:
                # Basic quality indicators
                width, height = img.size
                
                # Resolution score (higher is better, capped at 1.0)
                resolution_score = min(1.0, (width * height) / (1024 * 768))
                
                # Format score (JPEG/PNG preferred)
                format_score = 1.0 if img.format in ['JPEG', 'PNG'] else 0.7
                
                # Size score (not too small, not too large)
                size_score = 1.0 if 512 <= min(width, height) <= 2048 else 0.8
                
                # Combined quality score
                quality_score = (resolution_score + format_score + size_score) / 3.0
                
                return round(quality_score, 3)
                
        except Exception as e:
            logger.warning(f"Could not assess image quality for {image_path}: {e}")
            return 0.5  # Default moderate quality
    
    async def _create_training_session(self,
                                       model_version_id: int,
                                       session_id: str,
                                       config: Dict[str, Any] = None) -> None:
        """Create training session record"""
        
        if hasattr(self, '_mock_mode') and self._mock_mode:
            return
        
        insert_sql = """
        INSERT INTO training_sessions 
        (session_id, model_version_id, training_config, created_by)
        VALUES ($1, $2, $3, $4)
        """
        
        async with self.pg_pool.acquire() as conn:
            await conn.execute(
                insert_sql,
                session_id,
                model_version_id,
                json.dumps(config or {}),
                "training_pipeline"
            )
    
    async def _log_audit(self,
                         operation: str,
                         table_name: str,
                         record_id: int,
                         batman_token: str = None) -> None:
        """Log audit trail for compliance"""
        
        if hasattr(self, '_mock_mode') and self._mock_mode:
            logger.info(f"Mock audit: {operation} on {table_name}#{record_id}")
            return
        
        insert_sql = """
        INSERT INTO data_audit 
        (operation_type, table_name, record_id, batman_token, performed_by)
        VALUES ($1, $2, $3, $4, $5)
        """
        
        async with self.pg_pool.acquire() as conn:
            await conn.execute(
                insert_sql,
                operation,
                table_name,
                record_id,
                batman_token,
                "training_database"
            )
    
    async def _update_training_stats_cache(self) -> None:
        """Update Redis cache with training statistics"""
        if not self.redis_client:
            return
        
        try:
            # Get current dataset statistics
            stats = await self._get_dataset_statistics()
            
            # Cache for 1 hour
            await self.redis_client.setex(
                "vigia:training:stats",
                3600,
                json.dumps(stats, default=str)
            )
            
        except Exception as e:
            logger.warning(f"Failed to update training stats cache: {e}")
    
    async def _get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        
        if hasattr(self, '_mock_mode') and self._mock_mode:
            return {
                'total_images': 2500,
                'grade_distribution': {
                    '0': 800, '1': 600, '2': 500, '3': 300, '4': 200, 'U': 80, 'DTI': 20
                },
                'avg_quality_score': 0.87,
                'last_updated': datetime.now().isoformat()
            }
        
        query = """
        SELECT 
            COUNT(*) as total_images,
            AVG(image_quality_score) as avg_quality_score,
            COUNT(CASE WHEN npuap_grade = '0' THEN 1 END) as stage_0,
            COUNT(CASE WHEN npuap_grade = '1' THEN 1 END) as stage_1,
            COUNT(CASE WHEN npuap_grade = '2' THEN 1 END) as stage_2,
            COUNT(CASE WHEN npuap_grade = '3' THEN 1 END) as stage_3,
            COUNT(CASE WHEN npuap_grade = '4' THEN 1 END) as stage_4,
            COUNT(CASE WHEN npuap_grade = 'U' THEN 1 END) as unstageable,
            COUNT(CASE WHEN npuap_grade = 'DTI' THEN 1 END) as dti
        FROM medical_images
        """
        
        async with self.pg_pool.acquire() as conn:
            row = await conn.fetchrow(query)
            
            return {
                'total_images': row['total_images'],
                'avg_quality_score': float(row['avg_quality_score'] or 0),
                'grade_distribution': {
                    '0': row['stage_0'],
                    '1': row['stage_1'],
                    '2': row['stage_2'],
                    '3': row['stage_3'],
                    '4': row['stage_4'],
                    'U': row['unstageable'],
                    'DTI': row['dti']
                },
                'last_updated': datetime.now().isoformat()
            }
    
    async def close(self) -> None:
        """Close database connections"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("TrainingDatabase connections closed")


# Singleton instance for easy import
training_db = TrainingDatabase()

__all__ = ['TrainingDatabase', 'NPUAPGrade', 'training_db']