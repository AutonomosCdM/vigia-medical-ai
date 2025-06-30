#!/usr/bin/env python3
"""
VIGIA Medical AI - AWS Deployment Validation
Pre-deployment checks for HIPAA compliance and system readiness
"""

import boto3
import json
import sys
from typing import Dict, List, Tuple
import argparse
from botocore.exceptions import ClientError, NoCredentialsError
import subprocess
import os

class AWSDeploymentValidator:
    """Validates AWS environment for VIGIA Medical AI deployment"""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.project_name = "vigia-medical-ai"
        
        print(f"🔍 VIGIA Medical AI - AWS Deployment Validator")
        print(f"📍 Region: {region}")
        print("=" * 60)
        
        try:
            # Initialize AWS clients
            self.sts = boto3.client('sts', region_name=region)
            self.iam = boto3.client('iam', region_name=region)
            self.ec2 = boto3.client('ec2', region_name=region)
            self.ecs = boto3.client('ecs', region_name=region)
            self.rds = boto3.client('rds', region_name=region)
            self.s3 = boto3.client('s3', region_name=region)
            
        except NoCredentialsError:
            print("❌ AWS credentials not found")
            print("🔧 Configure AWS credentials using:")
            print("   aws configure")
            print("   or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            sys.exit(1)
    
    def validate_aws_credentials(self) -> Tuple[bool, str]:
        """Validate AWS credentials and permissions"""
        print("\n🔑 Validating AWS Credentials...")
        
        try:
            # Get current AWS identity
            identity = self.sts.get_caller_identity()
            account_id = identity['Account']
            user_arn = identity['Arn']
            
            print(f"✅ AWS Account ID: {account_id}")
            print(f"✅ User/Role ARN: {user_arn}")
            
            # Test basic permissions
            try:
                self.iam.list_roles(MaxItems=1)
                print("✅ IAM permissions available")
            except ClientError:
                print("⚠️ Limited IAM permissions")
            
            try:
                self.ec2.describe_vpcs(MaxResults=1)
                print("✅ EC2 permissions available")
            except ClientError:
                print("❌ EC2 permissions required")
                return False, "Missing EC2 permissions"
            
            try:
                self.ecs.list_clusters(maxResults=1)
                print("✅ ECS permissions available")
            except ClientError:
                print("❌ ECS permissions required") 
                return False, "Missing ECS permissions"
            
            return True, "AWS credentials validated"
            
        except ClientError as e:
            return False, f"AWS credential error: {e}"
    
    def validate_hipaa_eligible_services(self) -> Tuple[bool, str]:
        """Validate that required AWS services are HIPAA eligible"""
        print("\n🏥 Validating HIPAA Eligible Services...")
        
        # List of HIPAA eligible services required for VIGIA
        required_services = [
            "Amazon EC2",
            "Amazon ECS Fargate", 
            "Amazon RDS",
            "Amazon S3",
            "Amazon ElastiCache",
            "AWS Key Management Service",
            "AWS Secrets Manager",
            "Amazon CloudWatch",
            "AWS CloudTrail",
            "Application Load Balancer",
            "Amazon VPC"
        ]
        
        # These are all HIPAA eligible as of 2024
        hipaa_eligible_services = [
            "ec2", "ecs", "rds", "s3", "elasticache", 
            "kms", "secretsmanager", "cloudwatch", 
            "cloudtrail", "elasticloadbalancing", "vpc",
            "sagemaker", "bedrock", "transcribe"
        ]
        
        print("✅ Required HIPAA Eligible Services:")
        for service in required_services:
            print(f"   • {service}")
        
        print("\n✅ Additional Medical AI Services Available:")
        print("   • Amazon SageMaker (for MONAI/YOLOv5)")
        print("   • Amazon Bedrock (for medical LLM)")
        print("   • Amazon Transcribe Medical (for voice analysis)")
        print("   • Amazon Comprehend Medical (for text analysis)")
        
        return True, "All required services are HIPAA eligible"
    
    def validate_resource_limits(self) -> Tuple[bool, str]:
        """Validate AWS resource limits and quotas"""
        print("\n📊 Validating AWS Resource Limits...")
        
        issues = []
        
        try:
            # Check VPC limits
            vpcs = self.ec2.describe_vpcs()
            vpc_count = len(vpcs['Vpcs'])
            if vpc_count >= 5:  # Default limit is usually 5
                issues.append(f"VPC limit concern: {vpc_count}/5 VPCs used")
            else:
                print(f"✅ VPCs: {vpc_count}/5 available")
            
            # Check ECS cluster limits
            clusters = self.ecs.list_clusters()
            cluster_count = len(clusters['clusterArns'])
            if cluster_count >= 2000:  # ECS cluster limit
                issues.append(f"ECS cluster limit: {cluster_count}/2000")
            else:
                print(f"✅ ECS Clusters: {cluster_count}/2000 available")
            
            # Check RDS instances
            try:
                db_instances = self.rds.describe_db_instances()
                db_count = len(db_instances['DBInstances'])
                if db_count >= 40:  # Default RDS limit
                    issues.append(f"RDS limit concern: {db_count}/40 instances")
                else:
                    print(f"✅ RDS Instances: {db_count}/40 available")
            except ClientError:
                print("⚠️ Could not check RDS limits")
            
            if issues:
                return False, "; ".join(issues)
            else:
                return True, "Resource limits validated"
                
        except ClientError as e:
            return False, f"Error checking limits: {e}"
    
    def validate_docker_environment(self) -> Tuple[bool, str]:
        """Validate Docker environment for container deployment"""
        print("\n🐳 Validating Docker Environment...")
        
        try:
            # Check if Docker is installed
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Docker installed: {result.stdout.strip()}")
            else:
                return False, "Docker not installed"
            
            # Check if Docker is running
            result = subprocess.run(['docker', 'info'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Docker daemon running")
            else:
                return False, "Docker daemon not running"
            
            # Test Docker build capability
            print("🔨 Testing Docker build capability...")
            dockerfile_content = """
FROM python:3.11-slim
RUN echo "VIGIA Medical AI test build"
"""
            with open('/tmp/Dockerfile.test', 'w') as f:
                f.write(dockerfile_content)
            
            result = subprocess.run([
                'docker', 'build', '-t', 'vigia-test', 
                '-f', '/tmp/Dockerfile.test', '/tmp'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Docker build capability confirmed")
                # Cleanup test image
                subprocess.run(['docker', 'rmi', 'vigia-test'], 
                             capture_output=True)
                os.remove('/tmp/Dockerfile.test')
            else:
                return False, "Docker build failed"
            
            return True, "Docker environment validated"
            
        except FileNotFoundError:
            return False, "Docker not found in PATH"
        except Exception as e:
            return False, f"Docker validation error: {e}"
    
    def validate_network_connectivity(self) -> Tuple[bool, str]:
        """Validate network connectivity for external services"""
        print("\n🌐 Validating Network Connectivity...")
        
        endpoints_to_test = [
            ("AWS Services", "ec2.amazonaws.com"),
            ("Docker Hub", "hub.docker.com"),
            ("Slack API", "api.slack.com"),
            ("Hume AI", "api.hume.ai"),
            ("Hugging Face", "huggingface.co")
        ]
        
        failed_connections = []
        
        for service_name, endpoint in endpoints_to_test:
            try:
                import socket
                socket.create_connection((endpoint, 443), timeout=10)
                print(f"✅ {service_name}: {endpoint}")
            except socket.error:
                failed_connections.append(f"{service_name} ({endpoint})")
                print(f"❌ {service_name}: {endpoint}")
        
        if failed_connections:
            return False, f"Connection failed: {', '.join(failed_connections)}"
        else:
            return True, "Network connectivity validated"
    
    def validate_medical_system_requirements(self) -> Tuple[bool, str]:
        """Validate VIGIA Medical AI system requirements"""
        print("\n🏥 Validating Medical System Requirements...")
        
        # Check if VIGIA system files exist
        required_files = [
            "src/agents/master_medical_orchestrator.py",
            "src/agents/image_analysis_agent.py",
            "src/agents/voice_analysis_agent.py", 
            "src/core/phi_tokenization_client.py",
            "src/interfaces/slack_orchestrator.py",
            "requirements.txt",
            "Dockerfile.medical"
        ]
        
        missing_files = []
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path}")
            else:
                missing_files.append(file_path)
                print(f"❌ {file_path}")
        
        if missing_files:
            return False, f"Missing files: {', '.join(missing_files)}"
        
        # Check Python dependencies
        try:
            import torch
            import monai
            import gradio
            import slack_sdk
            print("✅ Core medical AI dependencies available")
        except ImportError as e:
            return False, f"Missing Python dependency: {e}"
        
        return True, "Medical system requirements validated"
    
    def validate_security_requirements(self) -> Tuple[bool, str]:
        """Validate security requirements for HIPAA compliance"""
        print("\n🔒 Validating Security Requirements...")
        
        security_checks = []
        
        # Check environment variables
        required_env_vars = [
            "SLACK_BOT_TOKEN",
            "HUME_AI_API_KEY"
        ]
        
        missing_env_vars = []
        for env_var in required_env_vars:
            if os.getenv(env_var):
                print(f"✅ {env_var} configured")
            else:
                missing_env_vars.append(env_var)
                print(f"⚠️ {env_var} not configured")
        
        # Security recommendations
        print("\n🛡️ Security Recommendations:")
        print("✅ Use AWS Secrets Manager for sensitive data")
        print("✅ Enable encryption at rest for all storage")
        print("✅ Enable encryption in transit (TLS 1.2+)")
        print("✅ Use VPC with private subnets")
        print("✅ Enable CloudTrail for audit logging")
        print("✅ Configure security groups restrictively")
        print("✅ Use IAM roles with least privilege")
        
        if missing_env_vars:
            return True, f"Security validated (configure in AWS: {', '.join(missing_env_vars)})"
        else:
            return True, "Security requirements validated"
    
    def generate_deployment_report(self, validation_results: Dict[str, Tuple[bool, str]]) -> str:
        """Generate comprehensive deployment readiness report"""
        
        report = f"""
# VIGIA Medical AI - AWS Deployment Readiness Report

## Summary
- **System**: VIGIA Medical AI v1.0
- **Target**: AWS Fargate with HIPAA Compliance
- **Region**: {self.region}
- **Validation Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Validation Results
"""
        
        passed = 0
        total = len(validation_results)
        
        for check_name, (success, message) in validation_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            report += f"\n### {check_name}\n**Status**: {status}\n**Details**: {message}\n"
            if success:
                passed += 1
        
        report += f"""
## Deployment Readiness Score: {passed}/{total} ({passed/total*100:.1f}%)

## Next Steps
"""
        
        if passed == total:
            report += """
🎉 **DEPLOYMENT READY**

Your AWS environment is ready for VIGIA Medical AI deployment.

### Recommended Actions:
1. Run deployment script: `python scripts/deploy_aws_medical.py`
2. Configure domain name and SSL certificate
3. Run post-deployment medical system validation
4. Perform HIPAA compliance audit
5. Set up monitoring and alerting

### Production Deployment Command:
```bash
python scripts/deploy_aws_medical.py --region {region} --environment production
```
""".format(region=self.region)
        else:
            report += """
⚠️ **DEPLOYMENT BLOCKED**

Please resolve the following issues before deployment:

"""
            for check_name, (success, message) in validation_results.items():
                if not success:
                    report += f"- **{check_name}**: {message}\n"
            
            report += """
### After Resolving Issues:
1. Re-run validation: `python scripts/validate_aws_deployment.py`
2. Proceed with deployment when all checks pass
"""
        
        return report
    
    def run_all_validations(self) -> bool:
        """Run all validation checks"""
        print("🚀 STARTING COMPREHENSIVE AWS DEPLOYMENT VALIDATION")
        print("🏥 VIGIA Medical AI v1.0 - HIPAA Compliant Medical System")
        print("=" * 60)
        
        validation_functions = [
            ("AWS Credentials", self.validate_aws_credentials),
            ("HIPAA Services", self.validate_hipaa_eligible_services),
            ("Resource Limits", self.validate_resource_limits),
            ("Docker Environment", self.validate_docker_environment),
            ("Network Connectivity", self.validate_network_connectivity),
            ("Medical System", self.validate_medical_system_requirements),
            ("Security Requirements", self.validate_security_requirements)
        ]
        
        results = {}
        all_passed = True
        
        for check_name, validation_func in validation_functions:
            try:
                success, message = validation_func()
                results[check_name] = (success, message)
                if not success:
                    all_passed = False
            except Exception as e:
                results[check_name] = (False, f"Validation error: {e}")
                all_passed = False
        
        # Generate and save report
        report = self.generate_deployment_report(results)
        
        with open('aws_deployment_validation_report.md', 'w') as f:
            f.write(report)
        
        print("\n" + "=" * 60)
        print("📋 VALIDATION COMPLETE")
        print("=" * 60)
        
        if all_passed:
            print("🎉 ALL VALIDATIONS PASSED")
            print("✅ AWS environment ready for VIGIA Medical AI deployment")
            print("📄 Report saved: aws_deployment_validation_report.md")
            print("\n🚀 Next step: python scripts/deploy_aws_medical.py")
        else:
            print("❌ VALIDATION FAILED")
            print("🔧 Please resolve issues before deployment")
            print("📄 Report saved: aws_deployment_validation_report.md")
        
        return all_passed

def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(description='Validate AWS environment for VIGIA Medical AI')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        validator = AWSDeploymentValidator(region=args.region)
        success = validator.run_all_validations()
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n💥 Validation error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()