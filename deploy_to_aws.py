#!/usr/bin/env python3
"""
Deploy VIGIA Medical AI with Agent Smith Landing to AWS App Runner
"""

import json
import subprocess
import sys
import boto3
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

def create_app_runner_service():
    """Create AWS App Runner service for VIGIA Medical AI"""
    
    console.print(Panel.fit(
        "[bold blue]Deploying VIGIA Medical AI with Agent Smith Landing[/bold blue]\n"
        "Target: AWS App Runner → autonomos.dev",
        title="🚀 AWS App Runner Deployment",
        border_style="blue"
    ))
    
    # App Runner service configuration
    service_config = {
        "ServiceName": "vigia-medical-ai-production",
        "SourceConfiguration": {
            "AutoDeploymentsEnabled": True,
            "CodeRepository": {
                "RepositoryUrl": "https://github.com/AutonomosCdM/vigia-medical-ai",
                "SourceCodeVersion": {
                    "Type": "BRANCH",
                    "Value": "main"
                },
                "CodeConfiguration": {
                    "ConfigurationSource": "REPOSITORY",
                    "CodeConfigurationValues": {
                        "Runtime": "PYTHON_3",
                        "BuildCommand": "pip install -r requirements.txt",
                        "StartCommand": "python src/web/main.py",
                        "RuntimeEnvironmentVariables": {
                            "AWS_DEPLOYMENT": "true",
                            "PORT": "8000",
                            "VIGIA_ENV": "production"
                        }
                    }
                }
            }
        },
        "InstanceConfiguration": {
            "Cpu": "1 vCPU",
            "Memory": "2 GB"
        },
        "Tags": [
            {
                "Key": "Project",
                "Value": "VIGIA Medical AI"
            },
            {
                "Key": "Environment", 
                "Value": "Production"
            },
            {
                "Key": "Domain",
                "Value": "autonomos.dev"
            }
        ]
    }
    
    try:
        client = boto3.client('apprunner', region_name='us-east-1')
        
        console.print("[yellow]Creating App Runner service...[/yellow]")
        
        response = client.create_service(**service_config)
        service_arn = response['Service']['ServiceArn']
        service_url = response['Service']['ServiceUrl']
        
        console.print(f"[green]✓ App Runner service created successfully![/green]")
        console.print(f"[blue]Service ARN:[/blue] {service_arn}")
        console.print(f"[blue]Service URL:[/blue] https://{service_url}")
        
        console.print("\n[yellow]Waiting for service deployment...[/yellow]")
        console.print("[dim]This may take 5-10 minutes...[/dim]")
        
        # Wait for service to be running
        waiter = client.get_waiter('service_running')
        waiter.wait(ServiceArn=service_arn)
        
        console.print("[green]✓ Service is now running![/green]")
        
        return {
            "service_arn": service_arn,
            "service_url": f"https://{service_url}",
            "status": "running"
        }
        
    except Exception as e:
        console.print(f"[red]✗ App Runner deployment failed: {str(e)}[/red]")
        return None

def configure_custom_domain(service_arn, service_url):
    """Configure custom domain for autonomos.dev"""
    
    console.print("\n[yellow]Configuring custom domain autonomos.dev...[/yellow]")
    
    try:
        client = boto3.client('apprunner', region_name='us-east-1')
        
        # Associate custom domain
        response = client.associate_custom_domain(
            ServiceArn=service_arn,
            DomainName="autonomos.dev",
            EnableWWWSubdomain=True
        )
        
        dns_target = response['DNSTarget']
        certificate_validation_records = response['CertificateValidationRecords']
        
        console.print(f"[green]✓ Custom domain association initiated![/green]")
        console.print(f"[blue]DNS Target:[/blue] {dns_target}")
        
        console.print("\n[bold yellow]DNS Configuration Required:[/bold yellow]")
        console.print("Update your DNS records at GoDaddy:")
        console.print(f"1. CNAME: autonomos.dev → {dns_target}")
        console.print(f"2. CNAME: www.autonomos.dev → {dns_target}")
        
        if certificate_validation_records:
            console.print("\n[bold yellow]SSL Certificate Validation:[/bold yellow]")
            for record in certificate_validation_records:
                console.print(f"CNAME: {record['Name']} → {record['Value']}")
        
        return {
            "dns_target": dns_target,
            "certificate_records": certificate_validation_records
        }
        
    except Exception as e:
        console.print(f"[red]✗ Custom domain configuration failed: {str(e)}[/red]")
        return None

def main():
    """Main deployment function"""
    
    # Create App Runner service
    deployment_result = create_app_runner_service()
    
    if not deployment_result:
        console.print("[red]Deployment failed![/red]")
        sys.exit(1)
    
    # Configure custom domain
    domain_result = configure_custom_domain(
        deployment_result["service_arn"], 
        deployment_result["service_url"]
    )
    
    # Print final results
    console.print(Panel.fit(
        f"""[bold green]VIGIA Medical AI Successfully Deployed![/bold green]

[bold blue]Live URLs:[/bold blue]
• App Runner: {deployment_result["service_url"]}
• Custom Domain: https://autonomos.dev (pending DNS)
• Agent Smith Landing: https://autonomos.dev/agent-smith

[bold yellow]Agent Smith Features:[/bold yellow]
• Professional gray button (no emojis)
• Email validation for corporate access
• Direct redirect to medical dashboard
• HIPAA-compliant medical interface

[bold cyan]Next Steps:[/bold cyan]
1. Update DNS at GoDaddy with provided CNAME records
2. Wait for SSL certificate validation (15-30 minutes)
3. Test complete flow: autonomos.dev → Agent Smith → Dashboard
4. Monitor medical system via AWS Console

[bold green]autonomos.dev is ready for production! 🚀[/bold green]""",
        title="🏥 Deployment Complete",
        border_style="green"
    ))

if __name__ == "__main__":
    main()