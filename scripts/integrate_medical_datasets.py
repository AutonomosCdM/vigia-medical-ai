#!/usr/bin/env python3
"""
VIGIA Medical AI - Medical Dataset Integration Runner
====================================================

Production script to integrate medical datasets (HAM10000, SkinCAP, synthetic data)
into the VIGIA training database with HIPAA compliance.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.dataset_integration import MedicalDatasetIntegrator
from src.utils.secure_logger import SecureLogger

logger = SecureLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Integrate medical datasets into VIGIA training database"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["ham10000", "skincap", "pressure_injury_synthetic", "all"],
        default=["all"],
        help="Datasets to integrate (default: all)"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Sample size per dataset (default: full dataset)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/integrated_datasets",
        help="Output directory for integrated datasets"
    )
    
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)"
    )
    
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with small samples"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--save-results",
        type=str,
        default="dataset_integration_results.json",
        help="Save results to JSON file"
    )
    
    return parser.parse_args()

async def integrate_datasets(args) -> Dict[str, Any]:
    """Main integration function"""
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🩺 VIGIA Medical Dataset Integration")
    logger.info("=" * 50)
    logger.info(f"Target datasets: {args.datasets}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Test mode: {args.test_mode}")
    
    start_time = datetime.now()
    
    try:
        # Initialize integrator
        integrator = MedicalDatasetIntegrator(storage_path=args.output_dir)
        await integrator.initialize()
        
        # Determine sample sizes
        if args.test_mode:
            sample_sizes = {
                "ham10000": 50,
                "skincap": 50,
                "pressure_injury_synthetic": 100
            }
            logger.info("🧪 Running in TEST MODE with small samples")
        elif args.sample_size:
            sample_sizes = {
                "ham10000": args.sample_size,
                "skincap": args.sample_size,
                "pressure_injury_synthetic": args.sample_size
            }
        else:
            sample_sizes = {
                "ham10000": None,  # Full dataset
                "skincap": None,   # Full dataset
                "pressure_injury_synthetic": 5000  # Reasonable synthetic size
            }
        
        results = {}
        
        # Determine which datasets to integrate
        if "all" in args.datasets:
            datasets_to_integrate = ["ham10000", "skincap", "pressure_injury_synthetic"]
        else:
            datasets_to_integrate = args.datasets
        
        logger.info(f"Integrating {len(datasets_to_integrate)} datasets...")
        
        # Integrate each dataset
        for dataset_name in datasets_to_integrate:
            try:
                logger.info(f"🔄 Starting integration: {dataset_name}")
                
                sample_size = sample_sizes.get(dataset_name)
                result = await integrator.integrate_dataset(
                    dataset_name, 
                    sample_size=sample_size,
                    validation_split=args.validation_split
                )
                
                results[dataset_name] = result
                
                logger.info(f"✅ {dataset_name}: {result['total_samples']} samples integrated")
                logger.info(f"   Train: {result['train_samples']}, Val: {result['validation_samples']}")
                logger.info(f"   Time: {result['processing_time_seconds']:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Failed to integrate {dataset_name}: {e}")
                results[dataset_name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Generate summary
        processing_time = (datetime.now() - start_time).total_seconds()
        successful_integrations = [r for r in results.values() if r.get("status") == "completed"]
        total_samples = sum([r.get("total_samples", 0) for r in successful_integrations])
        
        summary = {
            "integration_summary": {
                "total_datasets_requested": len(datasets_to_integrate),
                "successful_integrations": len(successful_integrations),
                "failed_integrations": len(datasets_to_integrate) - len(successful_integrations),
                "total_samples_integrated": total_samples,
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat(),
                "test_mode": args.test_mode,
                "output_directory": args.output_dir
            },
            "dataset_results": results,
            "configuration": {
                "validation_split": args.validation_split,
                "sample_sizes": sample_sizes,
                "datasets_requested": args.datasets
            }
        }
        
        # Print summary
        logger.info("=" * 50)
        logger.info("🏆 INTEGRATION SUMMARY")
        logger.info(f"   Successful: {len(successful_integrations)}/{len(datasets_to_integrate)} datasets")
        logger.info(f"   Total samples: {total_samples:,}")
        logger.info(f"   Processing time: {processing_time:.2f} seconds")
        
        if len(successful_integrations) == len(datasets_to_integrate):
            logger.info("🎉 All datasets integrated successfully!")
        else:
            logger.warning("⚠️  Some datasets failed to integrate. Check the logs above.")
        
        return summary
        
    except Exception as e:
        logger.error(f"Integration failed with error: {e}")
        raise

async def main():
    """Main function"""
    args = parse_arguments()
    
    try:
        # Run integration
        results = await integrate_datasets(args)
        
        # Save results
        if args.save_results:
            results_path = Path(args.save_results)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"📊 Results saved to: {results_path}")
        
        # Exit based on success
        successful = results["integration_summary"]["successful_integrations"]
        total = results["integration_summary"]["total_datasets_requested"]
        
        if successful == total:
            logger.info("✅ Integration completed successfully")
            sys.exit(0)
        else:
            logger.error(f"❌ Only {successful}/{total} datasets integrated successfully")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Integration cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Integration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Enable asyncio debug mode for development
    if "--verbose" in sys.argv:
        import os
        os.environ["PYTHONDEVMODE"] = "1"
    
    asyncio.run(main())