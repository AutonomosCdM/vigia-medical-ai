#!/usr/bin/env python3
"""
Setup Custom Domain for VIGIA Medical AI
======================================

Configure vigia.autonomos.dev to point to API Gateway
"""

import boto3
import json
from time import sleep

def setup_custom_domain():
    """Setup custom domain vigia.autonomos.dev for VIGIA Medical AI"""
    
    # AWS clients
    apigateway = boto3.client('apigateway', region_name='us-east-1')
    acm = boto3.client('acm', region_name='us-east-1')
    route53 = boto3.client('route53')
    
    print("🚀 Setting up vigia.autonomos.dev custom domain...")
    
    # Step 1: Request SSL Certificate
    print("📜 Requesting SSL certificate for vigia.autonomos.dev...")
    
    try:
        cert_response = acm.request_certificate(
            DomainName='vigia.autonomos.dev',
            ValidationMethod='DNS',
            SubjectAlternativeNames=[
                '*.vigia.autonomos.dev'
            ]
        )
        
        cert_arn = cert_response['CertificateArn']
        print(f"✅ Certificate requested: {cert_arn}")
        
        # Wait for certificate details
        print("⏳ Waiting for certificate validation details...")
        sleep(10)
        
        # Get certificate details for DNS validation
        cert_details = acm.describe_certificate(CertificateArn=cert_arn)
        
        if 'DomainValidationOptions' in cert_details['Certificate']:
            for validation in cert_details['Certificate']['DomainValidationOptions']:
                if 'ResourceRecord' in validation:
                    print(f"📋 DNS Validation Required:")
                    print(f"   Name: {validation['ResourceRecord']['Name']}")
                    print(f"   Value: {validation['ResourceRecord']['Value']}")
                    print(f"   Type: {validation['ResourceRecord']['Type']}")
        
    except Exception as e:
        print(f"❌ Certificate request failed: {e}")
        return
    
    # Step 2: Find API Gateway REST API ID
    print("🔍 Finding API Gateway for VIGIA...")
    
    try:
        apis = apigateway.get_rest_apis()
        vigia_api = None
        
        for api in apis['items']:
            if 'vigia' in api['name'].lower() or 'medical' in api['name'].lower():
                vigia_api = api
                break
        
        if not vigia_api:
            print("❌ VIGIA API Gateway not found")
            return
        
        api_id = vigia_api['id']
        print(f"✅ Found VIGIA API: {api_id}")
        
    except Exception as e:
        print(f"❌ API Gateway lookup failed: {e}")
        return
    
    # Step 3: Create custom domain (after certificate validation)
    print("🏗️  Custom domain creation script ready!")
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Add DNS validation record to GoDaddy:")
    print("   - Go to your GoDaddy DNS management")
    print("   - Add the CNAME record shown above")
    print("   - Wait for validation (5-30 minutes)")
    print("\n2. Run this command after validation:")
    print(f"   aws apigateway create-domain-name \\")
    print(f"     --domain-name vigia.autonomos.dev \\")
    print(f"     --certificate-arn {cert_arn} \\")
    print(f"     --region us-east-1")
    print("\n3. Create base path mapping:")
    print(f"   aws apigateway create-base-path-mapping \\")
    print(f"     --domain-name vigia.autonomos.dev \\")
    print(f"     --rest-api-id {api_id} \\")
    print(f"     --stage prod \\")
    print(f"     --region us-east-1")
    print("\n4. Add CNAME to GoDaddy DNS:")
    print("   Type: CNAME")
    print("   Name: vigia")
    print("   Data: [domain name from step 2 output]")

def check_certificate_status():
    """Check status of SSL certificates"""
    acm = boto3.client('acm', region_name='us-east-1')
    
    try:
        certs = acm.list_certificates()
        
        print("📜 SSL Certificates:")
        for cert in certs['CertificateSummary']:
            details = acm.describe_certificate(CertificateArn=cert['CertificateArn'])
            cert_info = details['Certificate']
            
            print(f"Domain: {cert_info['DomainName']}")
            print(f"Status: {cert_info['Status']}")
            print(f"ARN: {cert['CertificateArn']}")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Certificate check failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_certificate_status()
    else:
        setup_custom_domain()