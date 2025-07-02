#!/usr/bin/env python3
"""
VIGIA Medical AI Deployment Script
Deploys the complete VIGIA 9-Agent Medical System to AWS using Lambda + Step Functions
"""

import subprocess
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import boto3

console = Console()

def check_prerequisites():
    """Check if all prerequisites are installed"""
    
    console.print(Panel.fit(
        "[bold blue]VIGIA Medical AI Deployment - Prerequisites Check[/bold blue]",
        title="🔍 Checking Prerequisites"
    ))
    
    prerequisites = []
    
    # Check AWS CLI
    try:
        result = subprocess.run(['aws', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            console.print("[green]✓ AWS CLI installed[/green]")
        else:
            prerequisites.append("AWS CLI")
    except FileNotFoundError:
        prerequisites.append("AWS CLI")
        console.print("[red]✗ AWS CLI not found[/red]")
    
    # Check Node.js (required for CDK)
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            console.print("[green]✓ Node.js installed[/green]")
        else:
            prerequisites.append("Node.js")
    except FileNotFoundError:
        prerequisites.append("Node.js")
        console.print("[red]✗ Node.js not found[/red]")
    
    # Check CDK
    try:
        result = subprocess.run(['cdk', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            console.print("[green]✓ AWS CDK installed[/green]")
        else:
            prerequisites.append("AWS CDK")
    except FileNotFoundError:
        prerequisites.append("AWS CDK")
        console.print("[red]✗ AWS CDK not found[/red]")
    
    # Check AWS credentials
    try:
        boto3.Session().get_credentials()
        console.print("[green]✓ AWS credentials configured[/green]")
    except Exception as e:
        prerequisites.append("AWS credentials")
        console.print("[red]✗ AWS credentials not configured[/red]")
    
    if prerequisites:
        console.print(f"\\n[red]Missing prerequisites: {', '.join(prerequisites)}[/red]")
        console.print("\\nPlease install missing prerequisites:")
        console.print("1. AWS CLI: https://aws.amazon.com/cli/")
        console.print("2. Node.js: https://nodejs.org/")
        console.print("3. AWS CDK: npm install -g aws-cdk")
        console.print("4. AWS credentials: aws configure")
        return False
    
    console.print("\\n[bold green]✓ All prerequisites satisfied![/bold green]")
    return True

def install_dependencies():
    """Install Python dependencies"""
    
    console.print("\\n[yellow]Installing Python dependencies...[/yellow]")
    
    # Install CDK requirements
    infrastructure_dir = Path("infrastructure")
    if infrastructure_dir.exists():
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", 
            str(infrastructure_dir / "requirements.txt")
        ], check=True)
        console.print("[green]✓ CDK dependencies installed[/green]")
    
    # Install Lambda dependencies for each function
    lambda_dirs = [
        "lambda/master_orchestrator",
        "lambda/image_analysis",
        "lambda/voice_analysis", 
        "lambda/clinical_assessment",
        "lambda/risk_assessment",
        "lambda/diagnostic",
        "lambda/protocol",
        "lambda/communication",
        "lambda/monai_review"
    ]
    
    for lambda_dir in lambda_dirs:
        lambda_path = Path(lambda_dir)
        if lambda_path.exists():
            # Create requirements.txt for Lambda if it doesn't exist
            req_file = lambda_path / "requirements.txt"
            if not req_file.exists():
                with open(req_file, 'w') as f:
                    f.write("boto3>=1.26.0\\nnumpy>=1.21.0\\n")
            
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(req_file),
                "-t", str(lambda_path)
            ], check=True)
            console.print(f"[green]✓ {lambda_dir} dependencies installed[/green]")

def bootstrap_cdk():
    """Bootstrap CDK environment"""
    
    console.print("\\n[yellow]Bootstrapping CDK environment...[/yellow]")
    
    try:
        result = subprocess.run([
            "cdk", "bootstrap"
        ], capture_output=True, text=True, cwd="infrastructure")
        
        if result.returncode == 0:
            console.print("[green]✓ CDK environment bootstrapped[/green]")
        else:
            console.print(f"[red]✗ CDK bootstrap failed: {result.stderr}[/red]")
            return False
    except Exception as e:
        console.print(f"[red]✗ CDK bootstrap error: {str(e)}[/red]")
        return False
    
    return True

def synthesize_stack():
    """Synthesize CDK stack"""
    
    console.print("\\n[yellow]Synthesizing CDK stack...[/yellow]")
    
    try:
        result = subprocess.run([
            "cdk", "synth"
        ], capture_output=True, text=True, cwd="infrastructure")
        
        if result.returncode == 0:
            console.print("[green]✓ CDK stack synthesized successfully[/green]")
            return True
        else:
            console.print(f"[red]✗ CDK synth failed: {result.stderr}[/red]")
            return False
    except Exception as e:
        console.print(f"[red]✗ CDK synth error: {str(e)}[/red]")
        return False

def deploy_stack():
    """Deploy CDK stack to AWS"""
    
    console.print("\\n[yellow]Deploying VIGIA Medical AI stack to AWS...[/yellow]")
    console.print("[dim]This may take 15-20 minutes...[/dim]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Deploying VIGIA 9-Agent Medical System...", total=None)
            
            result = subprocess.run([
                "cdk", "deploy", "--require-approval", "never"
            ], capture_output=True, text=True, cwd="infrastructure")
            
            progress.remove_task(task)
        
        if result.returncode == 0:
            console.print("[bold green]✓ VIGIA Medical AI stack deployed successfully![/bold green]")
            
            # Extract outputs from deployment
            output_lines = result.stdout.split('\\n')
            outputs = {}
            for line in output_lines:
                if ' = ' in line and 'Output' in line:
                    parts = line.split(' = ')
                    if len(parts) == 2:
                        key = parts[0].split('.')[-1]
                        value = parts[1]
                        outputs[key] = value
            
            if outputs:
                console.print("\\n[bold blue]Deployment Outputs:[/bold blue]")
                for key, value in outputs.items():
                    console.print(f"  {key}: {value}")
            
            return True
        else:
            console.print(f"[red]✗ CDK deploy failed: {result.stderr}[/red]")
            return False
    except Exception as e:
        console.print(f"[red]✗ CDK deploy error: {str(e)}[/red]")
        return False

def run_post_deployment_tests():
    """Run basic tests to verify deployment"""
    
    console.print("\\n[yellow]Running post-deployment tests...[/yellow]")
    
    try:
        # Test Lambda functions exist
        lambda_client = boto3.client('lambda')
        
        functions_to_check = [
            'vigia-master-orchestrator',
            'vigia-image-analysis',
            'vigia-voice-analysis',
            'vigia-clinical-assessment',
            'vigia-risk-assessment',
            'vigia-diagnostic',
            'vigia-protocol',
            'vigia-communication',
            'vigia-monai-review'
        ]
        
        for function_name in functions_to_check:
            try:
                lambda_client.get_function(FunctionName=function_name)
                console.print(f"[green]✓ Lambda function {function_name} deployed[/green]")
            except lambda_client.exceptions.ResourceNotFoundException:
                console.print(f"[red]✗ Lambda function {function_name} not found[/red]")
        
        # Test DynamoDB tables exist
        dynamodb = boto3.client('dynamodb')
        
        tables_to_check = [
            'vigia-agents-state',
            'vigia-medical-audit',
            'vigia-lpp-results'
        ]
        
        for table_name in tables_to_check:
            try:
                dynamodb.describe_table(TableName=table_name)
                console.print(f"[green]✓ DynamoDB table {table_name} created[/green]")
            except dynamodb.exceptions.ResourceNotFoundException:
                console.print(f"[red]✗ DynamoDB table {table_name} not found[/red]")
        
        # Test Step Functions state machine
        sfn_client = boto3.client('stepfunctions')
        
        try:
            state_machines = sfn_client.list_state_machines()
            vigia_workflow = None
            for sm in state_machines['stateMachines']:
                if 'vigia-9-agent-workflow' in sm['name']:
                    vigia_workflow = sm
                    break
            
            if vigia_workflow:
                console.print("[green]✓ Step Functions workflow deployed[/green]")
            else:
                console.print("[red]✗ Step Functions workflow not found[/red]")
        
        except Exception as e:
            console.print(f"[red]✗ Step Functions check failed: {str(e)}[/red]")
        
        console.print("\\n[bold green]✓ Post-deployment tests completed![/bold green]")
        
    except Exception as e:
        console.print(f"[red]✗ Post-deployment tests failed: {str(e)}[/red]")

def print_next_steps():
    """Print next steps after successful deployment"""
    
    console.print(Panel.fit(
        """[bold green]VIGIA Medical AI System Successfully Deployed![/bold green]

[bold blue]Next Steps for Medical AI Deployment:[/bold blue]

1. 🏥 Test medical workflow via API Gateway endpoint
2. 📊 Monitor agents in DynamoDB tables
3. 🩺 Upload sample medical images to S3 bucket
4. 🔬 Trigger 9-agent analysis pipeline
5. 📱 Setup WhatsApp/Slack integration
6. 📈 Configure medical dashboards

[bold yellow]Medical System Endpoints:[/bold yellow]
- API Gateway: Check deployment outputs above
- Step Functions Console: AWS Console > Step Functions
- DynamoDB Tables: AWS Console > DynamoDB
- Lambda Functions: AWS Console > Lambda
- Medical Storage: S3 Console

[bold cyan]VIGIA Medical AI v1.0[/bold cyan]
Ready for pressure injury detection with 9-agent coordination!""",
        title="🩺 Deployment Complete",
        border_style="green"
    ))

def main():
    """Main deployment function"""
    
    console.print(Panel.fit(
        "[bold blue]VIGIA Medical AI (Sistema de Detección de LPP)[/bold blue]\\n"
        "9-Agent Medical AI System Deployment\\n"
        "Target: Production-Ready Pressure Injury Detection",
        title="🏥 VIGIA Deployment",
        border_style="blue"
    ))
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Run deployment steps
    if not check_prerequisites():
        sys.exit(1)
    
    try:
        install_dependencies()
        
        if not bootstrap_cdk():
            sys.exit(1)
        
        if not synthesize_stack():
            sys.exit(1)
        
        if not deploy_stack():
            sys.exit(1)
        
        run_post_deployment_tests()
        print_next_steps()
        
    except KeyboardInterrupt:
        console.print("\\n[red]Deployment cancelled by user[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\\n[red]Deployment failed: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()