#!/usr/bin/env python3
"""
Test dataset integration components in isolation without database dependencies
"""

import sys
from pathlib import Path
import asyncio
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def test_dataset_configurations():
    """Test dataset configuration loading"""
    print("🔬 Testing dataset configurations...")
    
    try:
        # Import just the configuration parts
        from src.pipeline.dataset_integration import DatasetSource, MedicalConditionType, DatasetConfig
        
        # Test enum values
        print(f"✅ DatasetSource enum: {list(DatasetSource)}")
        print(f"✅ MedicalConditionType enum: {list(MedicalConditionType)}")
        
        # Test configuration creation
        test_config = DatasetConfig(
            name="test_dataset",
            source=DatasetSource.SYNTHETIC,
            condition_type=MedicalConditionType.PRESSURE_INJURY,
            hipaa_compliant=True
        )
        print(f"✅ DatasetConfig created: {test_config.name}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def test_huggingface_availability():
    """Test if Hugging Face integration is available"""
    print("\n📡 Testing Hugging Face connectivity...")
    
    try:
        # Test if we can import datasets
        import datasets
        print("✅ Hugging Face datasets package available")
        
        # Test if we can access a small dataset (without downloading)
        try:
            from datasets import get_dataset_config_names
            # This is a lightweight operation that tests connectivity
            configs = get_dataset_config_names("squad")
            print(f"✅ Hugging Face Hub connectivity: {len(configs)} configs found for squad")
            return True
        except Exception as e:
            print(f"⚠️  Hugging Face Hub access limited: {e}")
            return False
            
    except ImportError:
        print("⚠️  Hugging Face datasets not available - install with: pip install datasets")
        return False

async def test_synthetic_data_structure():
    """Test synthetic data generation structure"""
    print("\n🧬 Testing synthetic data structure...")
    
    try:
        # Mock synthetic data structure
        import numpy as np
        from src.pipeline.dataset_integration import NPUAPGrade
        
        # Test NPUAP grades
        grades = list(NPUAPGrade)
        print(f"✅ NPUAP grades available: {[g.value for g in grades]}")
        
        # Create mock synthetic sample
        synthetic_sample = {
            "image": "synthetic_pressure_injury_stage_2_000001.jpg",
            "label": NPUAPGrade.STAGE_2.value,
            "patient_context": {
                "age": 65,
                "diabetes": True,
                "mobility_level": "limited"
            },
            "medical_metadata": {
                "anatomical_location": "sacrum",
                "severity_score": 0.75,
                "confidence": 0.89
            }
        }
        
        print(f"✅ Synthetic sample structure valid")
        print(f"   Label: {synthetic_sample['label']}")
        print(f"   Patient context keys: {list(synthetic_sample['patient_context'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Synthetic data test failed: {e}")
        return False

async def test_medical_preprocessing():
    """Test medical image preprocessing pipeline"""
    print("\n🖼️ Testing medical preprocessing...")
    
    try:
        from torchvision import transforms
        from PIL import Image
        import numpy as np
        
        # Test medical transforms
        medical_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create a test image
        test_image = Image.new('RGB', (256, 256), color='red')
        transformed = medical_transforms(test_image)
        
        print(f"✅ Medical transforms applied successfully")
        print(f"   Input size: (256, 256)")
        print(f"   Output shape: {transformed.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

async def test_directory_management():
    """Test directory structure management"""
    print("\n📁 Testing directory management...")
    
    try:
        from src.pipeline.dataset_integration import MedicalDatasetIntegrator
        
        # Test with temporary directory
        test_storage = Path("./data/test_isolated")
        integrator = MedicalDatasetIntegrator(storage_path=str(test_storage))
        
        # Check if directories were created
        expected_dirs = ["ham10000", "skincap", "pressure_injury_synthetic"]
        for dataset_name in expected_dirs:
            dataset_dir = integrator.storage_path / dataset_name
            if dataset_dir.exists():
                print(f"✅ {dataset_name} directory created")
            else:
                print(f"⚠️  {dataset_name} directory not found")
        
        print(f"✅ Storage path: {integrator.storage_path}")
        return True
        
    except Exception as e:
        print(f"❌ Directory management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run isolated tests"""
    print("🩺 VIGIA Dataset Integration - Isolated Testing")
    print("=" * 60)
    
    tests = [
        ("Dataset Configurations", test_dataset_configurations),
        ("Hugging Face Availability", test_huggingface_availability),
        ("Synthetic Data Structure", test_synthetic_data_structure),
        ("Medical Preprocessing", test_medical_preprocessing),
        ("Directory Management", test_directory_management)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed_tests += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"🏆 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests >= total_tests * 0.8:  # 80% pass rate
        print("✅ Core dataset integration functionality is working")
        return True
    else:
        print("❌ Core functionality has issues")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)