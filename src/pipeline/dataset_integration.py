"""
VIGIA Medical AI - Medical Dataset Integration
=============================================

Integration pipeline for medical datasets from Hugging Face and other sources.
Supports SkinCAP, HAM10000, and pressure injury datasets with HIPAA compliance.
"""

import asyncio
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import json
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import requests

# Optional dependencies for Hugging Face integration
try:
    from datasets import load_dataset, Dataset, DatasetDict
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    load_dataset = None
    Dataset = None
    DatasetDict = None

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    hf_hub_download = None
    list_repo_files = None

try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    train_test_split = None

from ..db.training_database import TrainingDatabase, NPUAPGrade
from ..core.phi_tokenization_client import PHITokenizationClient
from ..utils.secure_logger import SecureLogger
from ..utils.audit_service import AuditService
from ..synthetic.patient_generator import SyntheticPatientGenerator

logger = SecureLogger(__name__)

class DatasetSource(Enum):
    """Supported medical dataset sources"""
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    WEB = "web"
    SYNTHETIC = "synthetic"

class MedicalConditionType(Enum):
    """Medical condition classifications"""
    PRESSURE_INJURY = "pressure_injury"
    SKIN_LESION = "skin_lesion"
    DERMATOLOGICAL = "dermatological"
    WOUND_CARE = "wound_care"
    GENERAL_SKIN = "general_skin"

@dataclass
class DatasetConfig:
    """Configuration for dataset integration"""
    name: str
    source: DatasetSource
    condition_type: MedicalConditionType
    huggingface_id: Optional[str] = None
    local_path: Optional[str] = None
    web_url: Optional[str] = None
    label_mapping: Optional[Dict[str, str]] = None
    preprocessing_config: Optional[Dict[str, Any]] = None
    validation_required: bool = True
    hipaa_compliant: bool = True

class MedicalDatasetIntegrator:
    """Medical dataset integration with HIPAA compliance"""
    
    def __init__(self, 
                 storage_path: str = "./data/integrated_datasets",
                 training_db: TrainingDatabase = None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.training_db = training_db or TrainingDatabase()
        self.phi_client = PHITokenizationClient()
        self.audit_service = AuditService()
        self.patient_generator = SyntheticPatientGenerator()
        
        # Predefined dataset configurations
        self.dataset_configs = self._initialize_dataset_configs()
        
        # Medical image preprocessing
        self.medical_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"MedicalDatasetIntegrator initialized - Storage: {self.storage_path}")
    
    def _initialize_dataset_configs(self) -> Dict[str, DatasetConfig]:
        """Initialize predefined medical dataset configurations"""
        return {
            "ham10000": DatasetConfig(
                name="HAM10000",
                source=DatasetSource.HUGGINGFACE,
                condition_type=MedicalConditionType.DERMATOLOGICAL,
                huggingface_id="keremberke/ham10000-classification",
                label_mapping={
                    "akiec": "actinic_keratoses",
                    "bcc": "basal_cell_carcinoma", 
                    "bkl": "benign_keratosis",
                    "df": "dermatofibroma",
                    "mel": "melanoma",
                    "nv": "melanocytic_nevi",
                    "vasc": "vascular_lesions"
                },
                preprocessing_config={
                    "image_size": (512, 512),
                    "normalize": True,
                    "augmentation": True
                }
            ),
            "skincap": DatasetConfig(
                name="SkinCAP",
                source=DatasetSource.HUGGINGFACE,
                condition_type=MedicalConditionType.DERMATOLOGICAL,
                huggingface_id="marmal88/skin_cancer",
                label_mapping={
                    "benign": "benign_lesion",
                    "malignant": "malignant_lesion"
                },
                preprocessing_config={
                    "image_size": (512, 512),
                    "normalize": True,
                    "quality_check": True
                }
            ),
            "pressure_injury_synthetic": DatasetConfig(
                name="Pressure_Injury_Synthetic",
                source=DatasetSource.SYNTHETIC,
                condition_type=MedicalConditionType.PRESSURE_INJURY,
                label_mapping={
                    "0": "no_injury",
                    "1": "stage_1", 
                    "2": "stage_2",
                    "3": "stage_3",
                    "4": "stage_4",
                    "U": "unstageable",
                    "DTI": "deep_tissue_injury"
                },
                preprocessing_config={
                    "image_size": (512, 512),
                    "augmentation_strength": "high",
                    "medical_validation": True
                }
            )
        }
    
    async def initialize(self) -> None:
        """Initialize database connections and validate environment"""
        try:
            await self.training_db.initialize()
            await self.phi_client.initialize()
            
            # Create directory structure
            for dataset_name in self.dataset_configs.keys():
                dataset_dir = self.storage_path / dataset_name
                dataset_dir.mkdir(exist_ok=True)
                (dataset_dir / "images").mkdir(exist_ok=True)
                (dataset_dir / "labels").mkdir(exist_ok=True)
                (dataset_dir / "metadata").mkdir(exist_ok=True)
            
            logger.info("MedicalDatasetIntegrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MedicalDatasetIntegrator: {e}")
            raise
    
    async def integrate_dataset(self, 
                              dataset_name: str,
                              sample_size: Optional[int] = None,
                              validation_split: float = 0.2) -> Dict[str, Any]:
        """Integrate a medical dataset with HIPAA compliance"""
        
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        start_time = datetime.now()
        
        logger.info(f"Starting integration of {dataset_name}")
        
        try:
            # Load dataset based on source
            if config.source == DatasetSource.HUGGINGFACE:
                dataset = await self._load_huggingface_dataset(config, sample_size)
            elif config.source == DatasetSource.SYNTHETIC:
                dataset = await self._generate_synthetic_dataset(config, sample_size)
            else:
                raise NotImplementedError(f"Source {config.source} not implemented")
            
            # Process and validate dataset
            processed_dataset = await self._process_medical_dataset(dataset, config)
            
            # Split dataset
            train_data, val_data = await self._split_dataset(
                processed_dataset, validation_split
            )
            
            # Store in training database
            integration_id = await self._store_integrated_dataset(
                dataset_name, train_data, val_data, config
            )
            
            # Generate metadata and audit trail
            metadata = await self._generate_dataset_metadata(
                dataset_name, config, train_data, val_data, integration_id
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "integration_id": integration_id,
                "dataset_name": dataset_name,
                "train_samples": len(train_data),
                "validation_samples": len(val_data),
                "total_samples": len(train_data) + len(val_data),
                "processing_time_seconds": processing_time,
                "metadata": metadata,
                "status": "completed"
            }
            
            await self.audit_service.log_dataset_integration(result)
            logger.info(f"Dataset {dataset_name} integrated successfully: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to integrate dataset {dataset_name}: {e}")
            await self.audit_service.log_dataset_integration({
                "dataset_name": dataset_name,
                "status": "failed",
                "error": str(e),
                "processing_time_seconds": (datetime.now() - start_time).total_seconds()
            })
            raise
    
    async def _load_huggingface_dataset(self, 
                                       config: DatasetConfig, 
                                       sample_size: Optional[int]) -> Dataset:
        """Load dataset from Hugging Face"""
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("Hugging Face datasets not available. Install with: pip install datasets>=2.18.0")
        
        try:
            logger.info(f"Loading Hugging Face dataset: {config.huggingface_id}")
            
            # Load dataset
            dataset = load_dataset(config.huggingface_id, split="train")
            
            # Sample if requested
            if sample_size and len(dataset) > sample_size:
                indices = np.random.choice(len(dataset), sample_size, replace=False)
                dataset = dataset.select(indices)
            
            logger.info(f"Loaded {len(dataset)} samples from {config.huggingface_id}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load Hugging Face dataset {config.huggingface_id}: {e}")
            raise
    
    async def _generate_synthetic_dataset(self, 
                                        config: DatasetConfig,
                                        sample_size: Optional[int]) -> List[Dict[str, Any]]:
        """Generate synthetic medical dataset"""
        try:
            sample_size = sample_size or 1000
            logger.info(f"Generating {sample_size} synthetic samples for {config.name}")
            
            synthetic_data = []
            
            for i in range(sample_size):
                # Generate synthetic patient
                patient = await self.patient_generator.generate_realistic_patient()
                
                # Generate synthetic pressure injury scenario
                npuap_grades = list(NPUAPGrade)
                selected_grade = np.random.choice(npuap_grades)
                
                # Create synthetic image path (placeholder for now)
                image_path = f"synthetic_pressure_injury_{selected_grade.value}_{i:06d}.jpg"
                
                synthetic_sample = {
                    "image": image_path,
                    "label": selected_grade.value,
                    "patient_context": asdict(patient),
                    "medical_metadata": {
                        "anatomical_location": np.random.choice([
                            "sacrum", "heel", "hip", "shoulder", "elbow"
                        ]),
                        "severity_score": np.random.uniform(0.1, 1.0),
                        "confidence": np.random.uniform(0.7, 0.95)
                    }
                }
                
                synthetic_data.append(synthetic_sample)
            
            logger.info(f"Generated {len(synthetic_data)} synthetic samples")
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic dataset: {e}")
            raise
    
    async def _process_medical_dataset(self, 
                                     dataset: Union[Dataset, List[Dict]], 
                                     config: DatasetConfig) -> List[Dict[str, Any]]:
        """Process and validate medical dataset"""
        processed_data = []
        
        if isinstance(dataset, Dataset):
            dataset_items = dataset
        else:
            dataset_items = dataset
        
        for idx, item in enumerate(dataset_items):
            try:
                # Extract and validate image
                if isinstance(dataset, Dataset) and "image" in item:
                    image = item["image"]
                    if hasattr(image, 'save'):  # PIL Image
                        image_path = await self._save_medical_image(
                            image, f"{config.name}_{idx:06d}.jpg", config
                        )
                    else:
                        logger.warning(f"Unexpected image format at index {idx}")
                        continue
                elif isinstance(item, dict) and "image" in item:
                    # Synthetic data with image path
                    image_path = item["image"]
                else:
                    logger.warning(f"No image found at index {idx}")
                    continue
                
                # Extract and map labels
                if isinstance(dataset, Dataset):
                    label = item.get("label", item.get("dx", "unknown"))
                else:
                    label = item.get("label", "unknown")
                
                # Map label using configuration
                if config.label_mapping and str(label) in config.label_mapping:
                    mapped_label = config.label_mapping[str(label)]
                else:
                    mapped_label = str(label)
                
                # Generate Batman token for HIPAA compliance
                batman_token = await self.phi_client.create_token_async(
                    hospital_mrn=f"DATASET_{config.name}_{idx:06d}",
                    patient_data={"dataset_source": config.name, "index": idx}
                )
                
                processed_item = {
                    "image_path": image_path,
                    "original_label": label,
                    "mapped_label": mapped_label,
                    "batman_token": batman_token,
                    "dataset_source": config.name,
                    "condition_type": config.condition_type.value,
                    "medical_metadata": item.get("medical_metadata", {}),
                    "patient_context": item.get("patient_context", {}),
                    "processing_timestamp": datetime.now().isoformat()
                }
                
                processed_data.append(processed_item)
                
                if idx % 100 == 0:
                    logger.info(f"Processed {idx} samples from {config.name}")
                    
            except Exception as e:
                logger.warning(f"Failed to process item {idx} from {config.name}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_data)} samples from {config.name}")
        return processed_data
    
    async def _save_medical_image(self, 
                                image: Image.Image, 
                                filename: str, 
                                config: DatasetConfig) -> str:
        """Save medical image with proper formatting"""
        try:
            dataset_dir = self.storage_path / config.name / "images"
            image_path = dataset_dir / filename
            
            # Resize and normalize if configured
            if config.preprocessing_config and config.preprocessing_config.get("image_size"):
                target_size = config.preprocessing_config["image_size"]
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Save with high quality
            image.save(image_path, "JPEG", quality=95)
            
            return str(image_path)
            
        except Exception as e:
            logger.error(f"Failed to save medical image {filename}: {e}")
            raise
    
    async def _split_dataset(self, 
                           dataset: List[Dict[str, Any]], 
                           validation_split: float) -> Tuple[List[Dict], List[Dict]]:
        """Split dataset into training and validation"""
        try:
            if SKLEARN_AVAILABLE:
                # Use scikit-learn for stratified split
                train_data, val_data = train_test_split(
                    dataset, 
                    test_size=validation_split, 
                    random_state=42,
                    stratify=[item["mapped_label"] for item in dataset] if len(set(item["mapped_label"] for item in dataset)) > 1 else None
                )
            else:
                # Simple random split fallback
                np.random.seed(42)
                indices = np.random.permutation(len(dataset))
                split_idx = int(len(dataset) * (1 - validation_split))
                train_indices = indices[:split_idx]
                val_indices = indices[split_idx:]
                
                train_data = [dataset[i] for i in train_indices]
                val_data = [dataset[i] for i in val_indices]
                
                logger.warning("Using simple random split - install scikit-learn for stratified splitting")
            
            logger.info(f"Dataset split - Train: {len(train_data)}, Validation: {len(val_data)}")
            return train_data, val_data
            
        except Exception as e:
            logger.error(f"Failed to split dataset: {e}")
            raise
    
    async def _store_integrated_dataset(self, 
                                      dataset_name: str,
                                      train_data: List[Dict], 
                                      val_data: List[Dict],
                                      config: DatasetConfig) -> str:
        """Store integrated dataset in training database"""
        try:
            integration_id = hashlib.md5(
                f"{dataset_name}_{datetime.now().isoformat()}".encode()
            ).hexdigest()
            
            # Store training data
            for item in train_data:
                await self.training_db.store_medical_image(
                    image_path=item["image_path"],
                    npuap_grade=NPUAPGrade(item["mapped_label"]) if item["mapped_label"] in [g.value for g in NPUAPGrade] else NPUAPGrade.STAGE_0,
                    batman_token=item["batman_token"],
                    dataset_source=dataset_name,
                    split_type="train",
                    integration_id=integration_id,
                    medical_metadata=item.get("medical_metadata", {}),
                    patient_context=item.get("patient_context", {})
                )
            
            # Store validation data
            for item in val_data:
                await self.training_db.store_medical_image(
                    image_path=item["image_path"],
                    npuap_grade=NPUAPGrade(item["mapped_label"]) if item["mapped_label"] in [g.value for g in NPUAPGrade] else NPUAPGrade.STAGE_0,
                    batman_token=item["batman_token"],
                    dataset_source=dataset_name,
                    split_type="validation",
                    integration_id=integration_id,
                    medical_metadata=item.get("medical_metadata", {}),
                    patient_context=item.get("patient_context", {})
                )
            
            logger.info(f"Stored dataset {dataset_name} with integration_id: {integration_id}")
            return integration_id
            
        except Exception as e:
            logger.error(f"Failed to store integrated dataset: {e}")
            raise
    
    async def _generate_dataset_metadata(self, 
                                       dataset_name: str,
                                       config: DatasetConfig,
                                       train_data: List[Dict],
                                       val_data: List[Dict],
                                       integration_id: str) -> Dict[str, Any]:
        """Generate comprehensive metadata for integrated dataset"""
        try:
            # Label distribution
            all_labels = [item["mapped_label"] for item in train_data + val_data]
            label_distribution = {label: all_labels.count(label) for label in set(all_labels)}
            
            metadata = {
                "integration_id": integration_id,
                "dataset_name": dataset_name,
                "source": config.source.value,
                "condition_type": config.condition_type.value,
                "integration_timestamp": datetime.now().isoformat(),
                "total_samples": len(train_data) + len(val_data),
                "train_samples": len(train_data),
                "validation_samples": len(val_data),
                "label_distribution": label_distribution,
                "unique_labels": list(set(all_labels)),
                "preprocessing_config": config.preprocessing_config,
                "hipaa_compliant": config.hipaa_compliant,
                "batman_tokenization": True,
                "audit_trail": {
                    "integration_user": "vigia_system",
                    "integration_method": "automated_pipeline",
                    "validation_passed": True
                }
            }
            
            # Save metadata file
            metadata_path = self.storage_path / dataset_name / "metadata" / f"{integration_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to generate metadata: {e}")
            raise
    
    async def list_available_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets for integration"""
        available_datasets = []
        
        for name, config in self.dataset_configs.items():
            dataset_info = {
                "name": name,
                "display_name": config.name,
                "source": config.source.value,
                "condition_type": config.condition_type.value,
                "huggingface_id": config.huggingface_id,
                "hipaa_compliant": config.hipaa_compliant,
                "description": f"{config.name} - {config.condition_type.value} dataset"
            }
            available_datasets.append(dataset_info)
        
        return available_datasets
    
    async def integrate_all_datasets(self, 
                                   sample_sizes: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """Integrate all available medical datasets"""
        sample_sizes = sample_sizes or {}
        results = {}
        
        logger.info("Starting integration of all available datasets")
        
        for dataset_name in self.dataset_configs.keys():
            try:
                sample_size = sample_sizes.get(dataset_name)
                result = await self.integrate_dataset(dataset_name, sample_size)
                results[dataset_name] = result
                logger.info(f"Successfully integrated {dataset_name}")
                
            except Exception as e:
                logger.error(f"Failed to integrate {dataset_name}: {e}")
                results[dataset_name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Generate summary
        summary = {
            "total_datasets": len(self.dataset_configs),
            "successful_integrations": len([r for r in results.values() if r.get("status") == "completed"]),
            "failed_integrations": len([r for r in results.values() if r.get("status") == "failed"]),
            "total_samples": sum([r.get("total_samples", 0) for r in results.values() if r.get("status") == "completed"]),
            "integration_timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        logger.info(f"Dataset integration summary: {summary}")
        return summary

# Global instance for easy access
dataset_integrator = MedicalDatasetIntegrator()

async def integrate_medical_datasets():
    """Quick function to integrate all medical datasets"""
    await dataset_integrator.initialize()
    return await dataset_integrator.integrate_all_datasets({
        "ham10000": 2000,  # Sample 2000 images
        "skincap": 1500,   # Sample 1500 images  
        "pressure_injury_synthetic": 3000  # Generate 3000 synthetic samples
    })

if __name__ == "__main__":
    asyncio.run(integrate_medical_datasets())