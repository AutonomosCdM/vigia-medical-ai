#!/usr/bin/env python3
"""
VIGIA Medical AI - CDK Application Entry Point
9-Agent Medical AI System for Pressure Injury Detection
"""

import aws_cdk as cdk
from vigia_stack import VigiaStack

app = cdk.App()

# Deploy VIGIA Medical AI Stack
VigiaStack(app, "VigiaStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-east-1"
    ),
    description="VIGIA Medical AI - 9-Agent Pressure Injury Detection System with HIPAA Compliance"
)

app.synth()