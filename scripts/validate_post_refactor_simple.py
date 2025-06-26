#!/usr/bin/env python3
"""
Post-Refactor Validation Script
==============================

Simple validation script for CI/CD pipeline to ensure system integrity
after refactoring and organizational changes.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

def validate_project_structure() -> Dict[str, Any]:
    """Validate core project structure exists."""
    required_dirs = [
        'vigia_detect',
        'docs',
        'tests',
        'scripts',
        'config',
        'docker'
    ]
    
    results = {'status': 'pass', 'issues': []}
    
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            results['issues'].append(f"Missing required directory: {dir_name}")
            results['status'] = 'fail'
    
    return {'check': 'project_structure', **results}

def validate_requirements_files() -> Dict[str, Any]:
    """Validate requirements files are accessible."""
    required_files = [
        'requirements.txt',
        'config/requirements.txt',
        'vigia_detect/requirements.txt'
    ]
    
    results = {'status': 'pass', 'issues': []}
    
    for file_path in required_files:
        if not Path(file_path).exists():
            results['issues'].append(f"Missing requirements file: {file_path}")
            results['status'] = 'fail'
    
    return {'check': 'requirements_files', **results}

def validate_docker_configs() -> Dict[str, Any]:
    """Validate Docker configurations exist."""
    docker_files = [
        'docker/docker-compose.yml',
        'docker/docker-compose.hospital.yml',
        'docker/Dockerfile'
    ]
    
    results = {'status': 'pass', 'issues': []}
    
    for file_path in docker_files:
        if not Path(file_path).exists():
            results['issues'].append(f"Missing Docker config: {file_path}")
            results['status'] = 'fail'
    
    return {'check': 'docker_configs', **results}

def validate_core_modules() -> Dict[str, Any]:
    """Validate core Python modules can be imported."""
    modules_to_test = [
        'vigia_detect.core',
        'vigia_detect.cv_pipeline',
        'vigia_detect.utils'
    ]
    
    results = {'status': 'pass', 'issues': []}
    
    # Add project root to path
    sys.path.insert(0, str(Path.cwd()))
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
        except ImportError as e:
            results['issues'].append(f"Cannot import {module_name}: {e}")
            results['status'] = 'fail'
        except Exception as e:
            results['issues'].append(f"Error importing {module_name}: {e}")
            results['status'] = 'warning'
    
    return {'check': 'core_modules', **results}

def validate_security_files() -> Dict[str, Any]:
    """Validate security and environment files."""
    security_files = [
        '.env.testing',
        '.gitignore'
    ]
    
    results = {'status': 'pass', 'issues': []}
    
    for file_path in security_files:
        if not Path(file_path).exists():
            results['issues'].append(f"Missing security file: {file_path}")
            results['status'] = 'fail'
    
    # Check for exposed secrets in git
    if Path('.env').exists():
        results['issues'].append("Found .env file - should be gitignored")
        results['status'] = 'warning'
    
    return {'check': 'security_files', **results}

def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Post-refactor validation')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output', '-o', help='Output file for results')
    args = parser.parse_args()
    
    print("🔍 Starting post-refactor validation...")
    
    # Run all validation checks
    validations = [
        validate_project_structure(),
        validate_requirements_files(),
        validate_docker_configs(),
        validate_core_modules(),
        validate_security_files()
    ]
    
    # Analyze results
    total_checks = len(validations)
    passed_checks = sum(1 for v in validations if v['status'] == 'pass')
    failed_checks = sum(1 for v in validations if v['status'] == 'fail')
    warning_checks = sum(1 for v in validations if v['status'] == 'warning')
    
    # Print summary
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"   Total checks: {total_checks}")
    print(f"   ✅ Passed: {passed_checks}")
    print(f"   ⚠️  Warnings: {warning_checks}")
    print(f"   ❌ Failed: {failed_checks}")
    
    # Print detailed results if verbose
    if args.verbose:
        print(f"\n📋 DETAILED RESULTS:")
        for validation in validations:
            status_icon = {'pass': '✅', 'warning': '⚠️', 'fail': '❌'}.get(validation['status'], '❓')
            print(f"\n{status_icon} {validation['check'].replace('_', ' ').title()}: {validation['status'].upper()}")
            
            if validation.get('issues'):
                for issue in validation['issues']:
                    print(f"   • {issue}")
    
    # Save results to file if requested
    if args.output:
        results_data = {
            'summary': {
                'total_checks': total_checks,
                'passed': passed_checks,
                'warnings': warning_checks,
                'failed': failed_checks
            },
            'validations': validations,
            'overall_status': 'pass' if failed_checks == 0 else 'fail'
        }
        
        with open(args.output, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    # Determine exit code
    if failed_checks > 0:
        print(f"\n❌ Validation failed with {failed_checks} critical issues")
        return 1
    elif warning_checks > 0:
        print(f"\n⚠️  Validation passed with {warning_checks} warnings")
        return 0
    else:
        print(f"\n✅ All validations passed successfully!")
        return 0

if __name__ == '__main__':
    sys.exit(main())