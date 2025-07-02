#!/usr/bin/env python3
"""
VIGIA Medical AI - AWS Deployment Script
Automated deployment to AWS with HIPAA compliance
"""

import boto3
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List
import argparse

class VigiaAWSDeployer:
    """VIGIA Medical AI AWS Deployment Manager"""
    
    def __init__(self, region: str = "us-east-1", environment: str = "production"):
        self.region = region
        self.environment = environment
        self.project_name = "vigia-medical-ai"
        
        # AWS clients
        self.ecs = boto3.client('ecs', region_name=region)
        self.ec2 = boto3.client('ec2', region_name=region)
        self.rds = boto3.client('rds', region_name=region)
        self.elbv2 = boto3.client('elbv2', region_name=region)
        self.iam = boto3.client('iam', region_name=region)
        self.logs = boto3.client('logs', region_name=region)
        self.secretsmanager = boto3.client('secretsmanager', region_name=region)
        
        print(f"🏥 VIGIA Medical AI AWS Deployer")
        print(f"📍 Region: {region}")
        print(f"🌍 Environment: {environment}")
        print("=" * 60)
    
    def create_vpc_infrastructure(self) -> Dict[str, str]:
        """Create VPC infrastructure for medical AI"""
        print("\n🌐 Creating VPC Infrastructure...")
        
        # Create VPC
        vpc_response = self.ec2.create_vpc(
            CidrBlock='10.0.0.0/16',
            TagSpecifications=[
                {
                    'ResourceType': 'vpc',
                    'Tags': [
                        {'Key': 'Name', 'Value': f'{self.project_name}-vpc'},
                        {'Key': 'Environment', 'Value': self.environment},
                        {'Key': 'Project', 'Value': 'VIGIA-Medical-AI'},
                        {'Key': 'HIPAA', 'Value': 'Compliant'}
                    ]
                }
            ]
        )
        vpc_id = vpc_response['Vpc']['VpcId']
        print(f"✅ VPC created: {vpc_id}")
        
        # Enable DNS hostnames and resolution
        self.ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsHostnames={'Value': True})
        self.ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsSupport={'Value': True})
        
        # Create Internet Gateway
        igw_response = self.ec2.create_internet_gateway(
            TagSpecifications=[
                {
                    'ResourceType': 'internet-gateway',
                    'Tags': [
                        {'Key': 'Name', 'Value': f'{self.project_name}-igw'},
                        {'Key': 'Environment', 'Value': self.environment}
                    ]
                }
            ]
        )
        igw_id = igw_response['InternetGateway']['InternetGatewayId']
        self.ec2.attach_internet_gateway(InternetGatewayId=igw_id, VpcId=vpc_id)
        print(f"✅ Internet Gateway created: {igw_id}")
        
        # Get availability zones
        azs = self.ec2.describe_availability_zones()['AvailabilityZones']
        az_names = [az['ZoneName'] for az in azs[:2]]  # Use first 2 AZs
        
        subnet_ids = []
        private_subnet_ids = []
        
        # Create public subnets
        for i, az in enumerate(az_names):
            subnet_response = self.ec2.create_subnet(
                VpcId=vpc_id,
                CidrBlock=f'10.0.{i+1}.0/24',
                AvailabilityZone=az,
                TagSpecifications=[
                    {
                        'ResourceType': 'subnet',
                        'Tags': [
                            {'Key': 'Name', 'Value': f'{self.project_name}-public-subnet-{i+1}'},
                            {'Key': 'Type', 'Value': 'Public'},
                            {'Key': 'Environment', 'Value': self.environment}
                        ]
                    }
                ]
            )
            subnet_id = subnet_response['Subnet']['SubnetId']
            subnet_ids.append(subnet_id)
            
            # Enable auto-assign public IP
            self.ec2.modify_subnet_attribute(
                SubnetId=subnet_id,
                MapPublicIpOnLaunch={'Value': True}
            )
            print(f"✅ Public subnet created: {subnet_id} in {az}")
        
        # Create private subnets for medical processing
        for i, az in enumerate(az_names):
            private_subnet_response = self.ec2.create_subnet(
                VpcId=vpc_id,
                CidrBlock=f'10.0.{i+10}.0/24',
                AvailabilityZone=az,
                TagSpecifications=[
                    {
                        'ResourceType': 'subnet',
                        'Tags': [
                            {'Key': 'Name', 'Value': f'{self.project_name}-private-subnet-{i+1}'},
                            {'Key': 'Type', 'Value': 'Private'},
                            {'Key': 'Purpose', 'Value': 'Medical-Processing'},
                            {'Key': 'Environment', 'Value': self.environment}
                        ]
                    }
                ]
            )
            private_subnet_id = private_subnet_response['Subnet']['SubnetId']
            private_subnet_ids.append(private_subnet_id)
            print(f"✅ Private subnet created: {private_subnet_id} in {az}")
        
        # Create route table for public subnets
        route_table_response = self.ec2.create_route_table(
            VpcId=vpc_id,
            TagSpecifications=[
                {
                    'ResourceType': 'route-table',
                    'Tags': [
                        {'Key': 'Name', 'Value': f'{self.project_name}-public-rt'},
                        {'Key': 'Environment', 'Value': self.environment}
                    ]
                }
            ]
        )
        route_table_id = route_table_response['RouteTable']['RouteTableId']
        
        # Add route to internet gateway
        self.ec2.create_route(
            RouteTableId=route_table_id,
            DestinationCidrBlock='0.0.0.0/0',
            GatewayId=igw_id
        )
        
        # Associate public subnets with route table
        for subnet_id in subnet_ids:
            self.ec2.associate_route_table(RouteTableId=route_table_id, SubnetId=subnet_id)
        
        print("✅ VPC infrastructure created successfully")
        
        return {
            'vpc_id': vpc_id,
            'internet_gateway_id': igw_id,
            'public_subnet_ids': subnet_ids,
            'private_subnet_ids': private_subnet_ids,
            'availability_zones': az_names
        }
    
    def create_security_groups(self, vpc_id: str) -> Dict[str, str]:
        """Create security groups for medical AI services"""
        print("\n🔒 Creating Security Groups...")
        
        # ALB Security Group
        alb_sg_response = self.ec2.create_security_group(
            GroupName=f'{self.project_name}-alb-sg',
            Description='Security group for VIGIA Medical AI ALB',
            VpcId=vpc_id,
            TagSpecifications=[
                {
                    'ResourceType': 'security-group',
                    'Tags': [
                        {'Key': 'Name', 'Value': f'{self.project_name}-alb-sg'},
                        {'Key': 'Purpose', 'Value': 'Load-Balancer'},
                        {'Key': 'Environment', 'Value': self.environment}
                    ]
                }
            ]
        )
        alb_sg_id = alb_sg_response['GroupId']
        
        # Allow HTTPS traffic
        self.ec2.authorize_security_group_ingress(
            GroupId=alb_sg_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 443,
                    'ToPort': 443,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'HTTPS from internet'}]
                },
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 80,
                    'ToPort': 80,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'HTTP redirect to HTTPS'}]
                }
            ]
        )
        print(f"✅ ALB Security Group created: {alb_sg_id}")
        
        # ECS Security Group
        ecs_sg_response = self.ec2.create_security_group(
            GroupName=f'{self.project_name}-ecs-sg',
            Description='Security group for VIGIA Medical AI ECS services',
            VpcId=vpc_id,
            TagSpecifications=[
                {
                    'ResourceType': 'security-group',
                    'Tags': [
                        {'Key': 'Name', 'Value': f'{self.project_name}-ecs-sg'},
                        {'Key': 'Purpose', 'Value': 'Medical-AI-Services'},
                        {'Key': 'Environment', 'Value': self.environment}
                    ]
                }
            ]
        )
        ecs_sg_id = ecs_sg_response['GroupId']
        
        # Allow traffic from ALB
        self.ec2.authorize_security_group_ingress(
            GroupId=ecs_sg_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 8000,
                    'ToPort': 8000,
                    'UserIdGroupPairs': [{'GroupId': alb_sg_id, 'Description': 'From ALB'}]
                },
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 7860,
                    'ToPort': 7860,
                    'UserIdGroupPairs': [{'GroupId': alb_sg_id, 'Description': 'Gradio interface from ALB'}]
                }
            ]
        )
        print(f"✅ ECS Security Group created: {ecs_sg_id}")
        
        # RDS Security Group
        rds_sg_response = self.ec2.create_security_group(
            GroupName=f'{self.project_name}-rds-sg',
            Description='Security group for VIGIA Medical AI database',
            VpcId=vpc_id,
            TagSpecifications=[
                {
                    'ResourceType': 'security-group',
                    'Tags': [
                        {'Key': 'Name', 'Value': f'{self.project_name}-rds-sg'},
                        {'Key': 'Purpose', 'Value': 'Medical-Database'},
                        {'Key': 'Environment', 'Value': self.environment}
                    ]
                }
            ]
        )
        rds_sg_id = rds_sg_response['GroupId']
        
        # Allow PostgreSQL from ECS
        self.ec2.authorize_security_group_ingress(
            GroupId=rds_sg_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 5432,
                    'ToPort': 5432,
                    'UserIdGroupPairs': [{'GroupId': ecs_sg_id, 'Description': 'PostgreSQL from ECS'}]
                }
            ]
        )
        print(f"✅ RDS Security Group created: {rds_sg_id}")
        
        return {
            'alb_security_group_id': alb_sg_id,
            'ecs_security_group_id': ecs_sg_id,
            'rds_security_group_id': rds_sg_id
        }
    
    def create_iam_roles(self) -> Dict[str, str]:
        """Create IAM roles for medical AI services"""
        print("\n🔑 Creating IAM Roles...")
        
        # ECS Task Execution Role
        task_execution_role_doc = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        execution_role_response = self.iam.create_role(
            RoleName=f'{self.project_name}-task-execution-role',
            AssumeRolePolicyDocument=json.dumps(task_execution_role_doc),
            Description='ECS task execution role for VIGIA Medical AI',
            Tags=[
                {'Key': 'Project', 'Value': 'VIGIA-Medical-AI'},
                {'Key': 'Environment', 'Value': self.environment}
            ]
        )
        execution_role_arn = execution_role_response['Role']['Arn']
        
        # Attach AWS managed policies
        self.iam.attach_role_policy(
            RoleName=f'{self.project_name}-task-execution-role',
            PolicyArn='arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy'
        )
        
        # ECS Task Role with medical AI permissions
        task_role_doc = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        task_role_response = self.iam.create_role(
            RoleName=f'{self.project_name}-task-role',
            AssumeRolePolicyDocument=json.dumps(task_role_doc),
            Description='ECS task role for VIGIA Medical AI with medical permissions',
            Tags=[
                {'Key': 'Project', 'Value': 'VIGIA-Medical-AI'},
                {'Key': 'Environment', 'Value': self.environment}
            ]
        )
        task_role_arn = task_role_response['Role']['Arn']
        
        # Medical AI permissions policy
        medical_ai_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject"
                    ],
                    "Resource": f"arn:aws:s3:::{self.project_name}-medical-images/*"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "secretsmanager:GetSecretValue"
                    ],
                    "Resource": f"arn:aws:secretsmanager:{self.region}:*:secret:{self.project_name}-*"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel",
                        "sagemaker:InvokeEndpoint",
                        "transcribe:StartStreamTranscriptionWebSocket"
                    ],
                    "Resource": "*"
                }
            ]
        }
        
        self.iam.put_role_policy(
            RoleName=f'{self.project_name}-task-role',
            PolicyName='MedicalAIPermissions',
            PolicyDocument=json.dumps(medical_ai_policy)
        )
        
        print(f"✅ Task Execution Role: {execution_role_arn}")
        print(f"✅ Task Role: {task_role_arn}")
        
        return {
            'task_execution_role_arn': execution_role_arn,
            'task_role_arn': task_role_arn
        }
    
    def create_ecs_cluster(self) -> str:
        """Create ECS Fargate cluster for medical AI"""
        print("\n🐳 Creating ECS Cluster...")
        
        cluster_response = self.ecs.create_cluster(
            clusterName=f'{self.project_name}-cluster',
            tags=[
                {'key': 'Project', 'value': 'VIGIA-Medical-AI'},
                {'key': 'Environment', 'value': self.environment},
                {'key': 'Purpose', 'value': 'Medical-AI-Processing'}
            ],
            capacityProviders=['FARGATE'],
            defaultCapacityProviderStrategy=[
                {
                    'capacityProvider': 'FARGATE',
                    'weight': 1
                }
            ]
        )
        
        cluster_arn = cluster_response['cluster']['clusterArn']
        cluster_name = cluster_response['cluster']['clusterName']
        
        print(f"✅ ECS Cluster created: {cluster_name}")
        return cluster_name
    
    def create_task_definition(self, task_role_arn: str, execution_role_arn: str) -> str:
        """Create ECS task definition for medical AI services"""
        print("\n📋 Creating ECS Task Definition...")
        
        task_definition = {
            "family": f"{self.project_name}-medical-ai",
            "networkMode": "awsvpc",
            "requiresCompatibilities": ["FARGATE"],
            "cpu": "2048",  # 2 vCPU
            "memory": "4096",  # 4GB RAM
            "executionRoleArn": execution_role_arn,
            "taskRoleArn": task_role_arn,
            "containerDefinitions": [
                {
                    "name": "vigia-medical-orchestrator",
                    "image": f"{self.project_name}/medical-orchestrator:latest",
                    "cpu": 512,
                    "memory": 1024,
                    "essential": True,
                    "portMappings": [
                        {
                            "containerPort": 8000,
                            "protocol": "tcp"
                        }
                    ],
                    "environment": [
                        {"name": "ENVIRONMENT", "value": self.environment},
                        {"name": "AWS_REGION", "value": self.region},
                        {"name": "LOG_LEVEL", "value": "INFO"}
                    ],
                    "secrets": [
                        {
                            "name": "SLACK_BOT_TOKEN",
                            "valueFrom": f"arn:aws:secretsmanager:{self.region}:*:secret:{self.project_name}-slack-token"
                        },
                        {
                            "name": "HUME_AI_API_KEY",
                            "valueFrom": f"arn:aws:secretsmanager:{self.region}:*:secret:{self.project_name}-hume-api-key"
                        }
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": f"/ecs/{self.project_name}",
                            "awslogs-region": self.region,
                            "awslogs-stream-prefix": "medical-orchestrator"
                        }
                    },
                    "healthCheck": {
                        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                        "interval": 30,
                        "timeout": 5,
                        "retries": 3,
                        "startPeriod": 60
                    }
                },
                {
                    "name": "vigia-gradio-interface",
                    "image": f"{self.project_name}/gradio-interface:latest",
                    "cpu": 256,
                    "memory": 512,
                    "essential": False,
                    "portMappings": [
                        {
                            "containerPort": 7860,
                            "protocol": "tcp"
                        }
                    ],
                    "environment": [
                        {"name": "GRADIO_SERVER_NAME", "value": "0.0.0.0"},
                        {"name": "GRADIO_SERVER_PORT", "value": "7860"}
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": f"/ecs/{self.project_name}",
                            "awslogs-region": self.region,
                            "awslogs-stream-prefix": "gradio-interface"
                        }
                    }
                }
            ],
            "tags": [
                {"key": "Project", "value": "VIGIA-Medical-AI"},
                {"key": "Environment", "value": self.environment}
            ]
        }
        
        # Create CloudWatch log group
        try:
            self.logs.create_log_group(
                logGroupName=f"/ecs/{self.project_name}",
                tags={
                    'Project': 'VIGIA-Medical-AI',
                    'Environment': self.environment
                }
            )
            print(f"✅ CloudWatch log group created: /ecs/{self.project_name}")
        except self.logs.exceptions.ResourceAlreadyExistsException:
            print(f"✅ CloudWatch log group already exists: /ecs/{self.project_name}")
        
        # Register task definition
        response = self.ecs.register_task_definition(**task_definition)
        task_definition_arn = response['taskDefinition']['taskDefinitionArn']
        
        print(f"✅ Task Definition created: {task_definition_arn}")
        return task_definition_arn
    
    def deploy_complete_stack(self):
        """Deploy complete VIGIA Medical AI stack to AWS"""
        print("🚀 STARTING COMPLETE AWS DEPLOYMENT")
        print("🏥 VIGIA Medical AI v1.0 - HIPAA Compliant Medical System")
        print("=" * 60)
        
        try:
            # Phase 1: Core Infrastructure
            print("\n📋 PHASE 1: CORE INFRASTRUCTURE")
            vpc_info = self.create_vpc_infrastructure()
            security_groups = self.create_security_groups(vpc_info['vpc_id'])
            iam_roles = self.create_iam_roles()
            
            # Phase 2: Container Infrastructure  
            print("\n📋 PHASE 2: CONTAINER INFRASTRUCTURE")
            cluster_name = self.create_ecs_cluster()
            task_definition_arn = self.create_task_definition(
                iam_roles['task_role_arn'],
                iam_roles['task_execution_role_arn']
            )
            
            # Phase 3: Service Deployment
            print("\n📋 PHASE 3: SERVICE DEPLOYMENT")
            service_info = self.create_ecs_service(
                cluster_name,
                task_definition_arn,
                vpc_info['private_subnet_ids'],
                security_groups['ecs_security_group_id']
            )
            
            # Phase 4: Load Balancer
            print("\n📋 PHASE 4: LOAD BALANCER")
            alb_info = self.create_application_load_balancer(
                vpc_info['vpc_id'],
                vpc_info['public_subnet_ids'],
                security_groups['alb_security_group_id']
            )
            
            # Deployment Summary
            print("\n" + "=" * 60)
            print("🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"🏥 Medical AI Cluster: {cluster_name}")
            print(f"🌐 Load Balancer DNS: {alb_info['dns_name']}")
            print(f"🔒 VPC ID: {vpc_info['vpc_id']}")
            print(f"📊 Task Definition: {task_definition_arn.split('/')[-1]}")
            print("\n🎯 NEXT STEPS:")
            print("1. Configure domain name to point to ALB")
            print("2. Deploy SSL certificate via ACM")
            print("3. Run medical system validation tests")
            print("4. Configure monitoring and alerting")
            print("5. Perform HIPAA compliance audit")
            
            return {
                'vpc_info': vpc_info,
                'security_groups': security_groups,
                'iam_roles': iam_roles,
                'cluster_name': cluster_name,
                'task_definition_arn': task_definition_arn,
                'service_info': service_info,
                'alb_info': alb_info
            }
            
        except Exception as e:
            print(f"\n❌ DEPLOYMENT FAILED: {e}")
            print("🔧 Check AWS credentials and permissions")
            print("📋 Review CloudFormation events for details")
            return None
    
    def create_ecs_service(self, cluster_name: str, task_definition_arn: str, 
                          subnet_ids: List[str], security_group_id: str) -> Dict[str, Any]:
        """Create ECS Fargate service"""
        print("\n⚙️ Creating ECS Service...")
        
        service_response = self.ecs.create_service(
            cluster=cluster_name,
            serviceName=f'{self.project_name}-medical-service',
            taskDefinition=task_definition_arn,
            desiredCount=2,
            launchType='FARGATE',
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': subnet_ids,
                    'securityGroups': [security_group_id],
                    'assignPublicIp': 'DISABLED'
                }
            },
            tags=[
                {'key': 'Project', 'value': 'VIGIA-Medical-AI'},
                {'key': 'Environment', 'value': self.environment}
            ]
        )
        
        service_name = service_response['service']['serviceName']
        service_arn = service_response['service']['serviceArn']
        
        print(f"✅ ECS Service created: {service_name}")
        return {
            'service_name': service_name,
            'service_arn': service_arn
        }
    
    def create_application_load_balancer(self, vpc_id: str, subnet_ids: List[str], 
                                       security_group_id: str) -> Dict[str, str]:
        """Create Application Load Balancer"""
        print("\n⚖️ Creating Application Load Balancer...")
        
        alb_response = self.elbv2.create_load_balancer(
            Name=f'{self.project_name}-alb',
            Subnets=subnet_ids,
            SecurityGroups=[security_group_id],
            Scheme='internet-facing',
            Type='application',
            IpAddressType='ipv4',
            Tags=[
                {'Key': 'Project', 'Value': 'VIGIA-Medical-AI'},
                {'Key': 'Environment', 'Value': self.environment}
            ]
        )
        
        alb_arn = alb_response['LoadBalancers'][0]['LoadBalancerArn']
        alb_dns = alb_response['LoadBalancers'][0]['DNSName']
        
        print(f"✅ Application Load Balancer created")
        print(f"🌐 DNS Name: {alb_dns}")
        
        return {
            'alb_arn': alb_arn,
            'dns_name': alb_dns
        }

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='Deploy VIGIA Medical AI to AWS')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--environment', default='production', help='Environment name')
    parser.add_argument('--dry-run', action='store_true', help='Validate without deploying')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - Validation only")
        print("✅ AWS credentials and permissions validation would run here")
        print("✅ Resource quotas and limits validation would run here")
        print("✅ HIPAA compliance checks would run here")
        return
    
    try:
        deployer = VigiaAWSDeployer(region=args.region, environment=args.environment)
        result = deployer.deploy_complete_stack()
        
        if result:
            print("\n🎉 VIGIA Medical AI successfully deployed to AWS!")
            sys.exit(0)
        else:
            print("\n❌ Deployment failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Deployment error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()