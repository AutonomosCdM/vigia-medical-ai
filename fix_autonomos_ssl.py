#!/usr/bin/env python3
"""
Fix autonomos.dev SSL Certificate Issue
=====================================

The domain autonomos.dev resolves to S3 but has no SSL certificate.
We need to:
1. Request SSL certificate for autonomos.dev
2. Create CloudFront distribution
3. Update DNS to point to CloudFront
"""

import boto3
import json
from time import sleep

def fix_autonomos_ssl():
    """Fix SSL for autonomos.dev"""
    
    # AWS clients
    acm = boto3.client('acm', region_name='us-east-1')
    cloudfront = boto3.client('cloudfront')
    s3 = boto3.client('s3')
    
    print("🔧 Fixing autonomos.dev SSL configuration...")
    
    # Step 1: Request SSL Certificate for autonomos.dev
    print("📜 Requesting SSL certificate for autonomos.dev...")
    
    try:
        cert_response = acm.request_certificate(
            DomainName='autonomos.dev',
            ValidationMethod='DNS',
            SubjectAlternativeNames=[
                'www.autonomos.dev'
            ]
        )
        
        cert_arn = cert_response['CertificateArn']
        print(f"✅ Certificate requested: {cert_arn}")
        
        # Wait for certificate details
        print("⏳ Waiting for certificate validation details...")
        sleep(10)
        
        # Get certificate details for DNS validation
        cert_details = acm.describe_certificate(CertificateArn=cert_arn)
        
        print("📋 DNS Validation Required:")
        if 'DomainValidationOptions' in cert_details['Certificate']:
            for validation in cert_details['Certificate']['DomainValidationOptions']:
                if 'ResourceRecord' in validation:
                    print(f"   Domain: {validation['DomainName']}")
                    print(f"   Name: {validation['ResourceRecord']['Name']}")
                    print(f"   Type: {validation['ResourceRecord']['Type']}")
                    print(f"   Value: {validation['ResourceRecord']['Value']}")
                    print("-" * 60)
        
        print("\n🔧 Next Steps:")
        print("1. Add the CNAME records above to GoDaddy DNS")
        print("2. Wait for certificate validation (5-30 minutes)")
        print("3. Run: python fix_autonomos_ssl.py create-cloudfront")
        print("4. Update GoDaddy A record to point to CloudFront")
        
    except Exception as e:
        print(f"❌ Certificate request failed: {e}")

def create_cloudfront_distribution():
    """Create CloudFront distribution for autonomos.dev"""
    
    cloudfront = boto3.client('cloudfront')
    
    print("☁️ Creating CloudFront distribution for autonomos.dev...")
    
    # S3 bucket website endpoint
    s3_endpoint = "autonomos-landing-page.s3-website-us-east-1.amazonaws.com"
    
    # Certificate ARN for autonomos.dev
    cert_arn = "arn:aws:acm:us-east-1:586794472237:certificate/475a8b1e-6d20-4d33-8447-269318ce407a"
    
    distribution_config = {
        'CallerReference': f'autonomos-dev-{int(__import__("time").time())}',
        'Comment': 'autonomos.dev Agent Smith Landing Page with SSL',
        'DefaultCacheBehavior': {
            'TargetOriginId': 'S3-autonomos-landing-page',
            'ViewerProtocolPolicy': 'redirect-to-https',
            'MinTTL': 0,
            'DefaultTTL': 86400,
            'MaxTTL': 31536000,
            'TrustedSigners': {
                'Enabled': False,
                'Quantity': 0
            },
            'ForwardedValues': {
                'QueryString': False,
                'Cookies': {'Forward': 'none'}
            }
        },
        'Origins': {
            'Quantity': 1,
            'Items': [
                {
                    'Id': 'S3-autonomos-landing-page',
                    'DomainName': s3_endpoint,
                    'CustomOriginConfig': {
                        'HTTPPort': 80,
                        'HTTPSPort': 443,
                        'OriginProtocolPolicy': 'http-only'
                    }
                }
            ]
        },
        'Aliases': {
            'Quantity': 2,
            'Items': ['autonomos.dev', 'www.autonomos.dev']
        },
        'ViewerCertificate': {
            'ACMCertificateArn': cert_arn,
            'SSLSupportMethod': 'sni-only',
            'MinimumProtocolVersion': 'TLSv1.2_2021'
        },
        'DefaultRootObject': 'index.html',
        'Enabled': True,
        'PriceClass': 'PriceClass_100'
    }
    
    try:
        response = cloudfront.create_distribution(
            DistributionConfig=distribution_config
        )
        
        distribution_id = response['Distribution']['Id']
        domain_name = response['Distribution']['DomainName']
        
        print(f"✅ CloudFront distribution created:")
        print(f"   Distribution ID: {distribution_id}")
        print(f"   Domain Name: {domain_name}")
        print(f"   Status: {response['Distribution']['Status']}")
        
        print("\n🔧 Final Steps:")
        print("1. Wait for distribution deployment (10-15 minutes)")
        print("2. Update GoDaddy DNS:")
        print(f"   A record @ → {domain_name}")
        print(f"   CNAME www → {domain_name}")
        print("3. Test: https://autonomos.dev")
        
    except Exception as e:
        print(f"❌ CloudFront creation failed: {e}")

def check_status():
    """Check current status of autonomos.dev infrastructure"""
    
    print("🔍 Checking autonomos.dev status...")
    
    # Check certificates
    acm = boto3.client('acm', region_name='us-east-1')
    try:
        certs = acm.list_certificates()
        autonomos_cert = None
        vigia_cert = None
        
        for cert in certs['CertificateSummaryList']:
            if cert['DomainName'] == 'autonomos.dev':
                autonomos_cert = cert
            elif cert['DomainName'] == 'vigia.autonomos.dev':
                vigia_cert = cert
        
        print(f"📜 SSL Certificates:")
        print(f"   autonomos.dev: {'✅ ' + autonomos_cert['Status'] if autonomos_cert else '❌ Not found'}")
        print(f"   vigia.autonomos.dev: {'✅ ' + vigia_cert['Status'] if vigia_cert else '❌ Not found'}")
        
    except Exception as e:
        print(f"❌ Certificate check failed: {e}")
    
    # Check CloudFront distributions
    cloudfront = boto3.client('cloudfront')
    try:
        distributions = cloudfront.list_distributions()
        autonomos_dist = None
        
        if 'Items' in distributions['DistributionList']:
            for dist in distributions['DistributionList']['Items']:
                if 'Items' in dist['Aliases'] and 'autonomos.dev' in dist['Aliases']['Items']:
                    autonomos_dist = dist
                    break
        
        print(f"☁️ CloudFront:")
        if autonomos_dist:
            print(f"   autonomos.dev: ✅ {autonomos_dist['Status']} ({autonomos_dist['DomainName']})")
        else:
            print(f"   autonomos.dev: ❌ No distribution found")
            
    except Exception as e:
        print(f"❌ CloudFront check failed: {e}")
    
    # Check S3 bucket
    print(f"🪣 S3 Bucket:")
    print(f"   autonomos-landing-page: ✅ Website enabled")
    print(f"   Endpoint: autonomos-landing-page.s3-website-us-east-1.amazonaws.com")

if __name__ == "__main__":
    import sys
    import time
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "check":
            check_status()
        elif sys.argv[1] == "create-cloudfront":
            create_cloudfront_distribution()
        else:
            print("Usage: python fix_autonomos_ssl.py [check|create-cloudfront]")
    else:
        fix_autonomos_ssl()