#!/usr/bin/env python3
"""
Direct Hume AI API test with exact credentials from screenshot
"""
import requests
import json
import base64

# Exact API keys from screenshot
API_KEY = "qLuGEn89XUILGpTyAL9KYjXHXHfLuvRHbzgbbmHxJ6UOOi4Zq"
SECRET_KEY = "z6R7VLqyfwIbbIh46RFvxzCI93SbPyJMcnuRtfIE8tmv4WYAt3eYUruRSy7pFj6A"

def test_hume_api():
    """Test different authentication methods"""
    
    print("🔑 Testing Hume AI API with screenshot credentials")
    print(f"API Key: {API_KEY[:20]}...")
    print(f"Secret Key: {SECRET_KEY[:20]}...")
    print()
    
    # Method 1: X-Hume-Api-Key header
    print("Method 1: X-Hume-Api-Key header")
    headers = {
        'X-Hume-Api-Key': API_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(
            'https://api.hume.ai/v0/batch/jobs',
            headers=headers,
            timeout=10
        )
        print(f"📡 Status: {response.status_code}")
        print(f"📄 Response: {response.text[:200]}...")
        
        if response.status_code == 200:
            print("✅ Method 1 SUCCESS!")
            return True
    except Exception as e:
        print(f"❌ Method 1 Error: {e}")
    
    print()
    
    # Method 2: Basic Auth
    print("Method 2: Basic Authentication")
    auth_string = base64.b64encode(f"{API_KEY}:{SECRET_KEY}".encode()).decode()
    headers = {
        'Authorization': f'Basic {auth_string}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(
            'https://api.hume.ai/v0/batch/jobs',
            headers=headers,
            timeout=10
        )
        print(f"📡 Status: {response.status_code}")
        print(f"📄 Response: {response.text[:200]}...")
        
        if response.status_code == 200:
            print("✅ Method 2 SUCCESS!")
            return True
    except Exception as e:
        print(f"❌ Method 2 Error: {e}")
    
    print()
    
    # Method 3: Bearer token
    print("Method 3: Bearer Token")
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(
            'https://api.hume.ai/v0/batch/jobs',
            headers=headers,
            timeout=10
        )
        print(f"📡 Status: {response.status_code}")
        print(f"📄 Response: {response.text[:200]}...")
        
        if response.status_code == 200:
            print("✅ Method 3 SUCCESS!")
            return True
    except Exception as e:
        print(f"❌ Method 3 Error: {e}")
    
    print()
    
    # Method 4: API Key as parameter
    print("Method 4: API Key as URL parameter")
    try:
        response = requests.get(
            f'https://api.hume.ai/v0/batch/jobs?api_key={API_KEY}',
            timeout=10
        )
        print(f"📡 Status: {response.status_code}")
        print(f"📄 Response: {response.text[:200]}...")
        
        if response.status_code == 200:
            print("✅ Method 4 SUCCESS!")
            return True
    except Exception as e:
        print(f"❌ Method 4 Error: {e}")
    
    print()
    print("❌ All authentication methods failed")
    return False

if __name__ == "__main__":
    test_hume_api()