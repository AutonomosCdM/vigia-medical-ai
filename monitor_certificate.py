#!/usr/bin/env python3
"""
Monitor SSL Certificate Validation and Complete Domain Setup
===========================================================
"""

import boto3
import time
import sys

def monitor_certificate():
    """Monitor certificate validation and complete setup when ready"""
    
    cert_arn = "arn:aws:acm:us-east-1:586794472237:certificate/2e527767-c809-4122-85f0-91781e8b5125"
    api_id = "k0lhxhcl5a"
    
    acm = boto3.client('acm', region_name='us-east-1')
    apigateway = boto3.client('apigateway', region_name='us-east-1')
    
    print("🔍 Monitoring certificate validation...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            # Check certificate status
            response = acm.describe_certificate(CertificateArn=cert_arn)
            status = response['Certificate']['Status']
            
            print(f"⏰ {time.strftime('%H:%M:%S')} - Certificate Status: {status}")
            
            if status == 'ISSUED':
                print("🎉 Certificate validated! Creating custom domain...")
                
                # Create custom domain
                try:
                    domain_response = apigateway.create_domain_name(
                        domainName='vigia.autonomos.dev',
                        certificateArn=cert_arn
                    )
                    
                    domain_name_url = domain_response['distributionDomainName']
                    print(f"✅ Custom domain created!")
                    print(f"Domain URL: {domain_name_url}")
                    
                    # Create base path mapping
                    apigateway.create_base_path_mapping(
                        domainName='vigia.autonomos.dev',
                        restApiId=api_id,
                        stage='prod'
                    )
                    print("✅ Base path mapping created!")
                    
                    print("\n" + "="*60)
                    print("FINAL DNS SETUP REQUIRED:")
                    print("="*60)
                    print("Add this CNAME to GoDaddy DNS:")
                    print(f"Type: CNAME")
                    print(f"Name: vigia")
                    print(f"Data: {domain_name_url}")
                    print("\nAfter adding this, vigia.autonomos.dev will work!")
                    print("="*60)
                    
                    break
                    
                except Exception as e:
                    if "already exists" in str(e):
                        print("✅ Custom domain already exists, checking details...")
                        
                        # Get existing domain
                        domain_info = apigateway.get_domain_name(domainName='vigia.autonomos.dev')
                        domain_name_url = domain_info['distributionDomainName']
                        
                        print(f"✅ Domain URL: {domain_name_url}")
                        print("\n" + "="*60)
                        print("FINAL DNS SETUP REQUIRED:")
                        print("="*60)
                        print("Add this CNAME to GoDaddy DNS:")
                        print(f"Type: CNAME")
                        print(f"Name: vigia")
                        print(f"Data: {domain_name_url}")
                        print("="*60)
                        break
                    else:
                        print(f"❌ Error creating domain: {e}")
                        break
            
            elif status == 'FAILED':
                print("❌ Certificate validation failed!")
                print("Check your DNS configuration and try again.")
                break
            
            # Wait 30 seconds before checking again
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")
        print("Run again anytime to check status.")

if __name__ == "__main__":
    monitor_certificate()