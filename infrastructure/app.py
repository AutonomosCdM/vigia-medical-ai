#!/usr/bin/env python3
"""
VIGIA Medical AI - CDK Application Entry Point
9-Agent Medical AI System for Pressure Injury Detection
"""

import aws_cdk as cdk
from vigia_stack import VigiaStack

app = cdk.App()

# Deploy VIGIA Medical AI Stack with explicit environment for custom domain
VigiaStack(app, "VigiaStack",
    env=cdk.Environment(
        account="586794472237",  # Autonomos AI Lab AWS account
        region="us-east-1"       # Primary deployment region
    ),
    description="VIGIA Medical AI - 9-Agent Pressure Injury Detection System with HIPAA Compliance"
)

app.synth()