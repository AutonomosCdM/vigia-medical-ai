#!/usr/bin/env python3
"""
Test Hume AI with correct API format based on documentation
"""
import requests
import json

# From screenshot - let me re-read the exact values
API_KEY = "qluGEn89XUILGpTyAL9KYjXHKHFLuvRHbzgbbmHxJ6UOOi4Zq"
SECRET_KEY = "z6R7VLqyfwIbbIh46RFvxzCI93SbPyJMcnuStfIE8tmv4WYAt3eYUruRSy7pFj6A"

def test_hume_correct():
    """Test with correct Hume AI format"""
    
    print("🔑 Testing Hume AI with corrected API format")
    print(f"API Key: {API_KEY[:20]}...")
    print()
    
    # Test 1: Expression Measurement API (correct endpoint)
    print("Test 1: Expression Measurement API")
    headers = {
        'X-Hume-Api-Key': API_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        # Try the correct endpoint for expression measurement
        response = requests.get(
            'https://api.hume.ai/v0/batch/models',
            headers=headers,
            timeout=10
        )
        print(f"📡 Status: {response.status_code}")
        print(f"📄 Response: {response.text[:300]}...")
        
        if response.status_code == 200:
            print("✅ SUCCESS! API is working")
            return True
        elif response.status_code == 401:
            print("❌ Authentication failed - API key invalid")
        else:
            print(f"❌ Unexpected status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    
    # Test 2: Try to get token for access
    print("Test 2: Token authentication")
    try:
        token_url = "https://api.hume.ai/oauth2-cc/token"
        token_data = {
            'grant_type': 'client_credentials',
            'client_id': API_KEY,
            'client_secret': SECRET_KEY
        }
        
        response = requests.post(
            token_url,
            data=token_data,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=10
        )
        
        print(f"📡 Token Status: {response.status_code}")
        print(f"📄 Token Response: {response.text[:300]}...")
        
        if response.status_code == 200:
            token_info = response.json()
            access_token = token_info.get('access_token')
            print(f"✅ Got access token: {access_token[:20]}...")
            
            # Now try with the access token
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(
                'https://api.hume.ai/v0/batch/models',
                headers=headers,
                timeout=10
            )
            
            print(f"📡 Authenticated Status: {response.status_code}")
            print(f"📄 Authenticated Response: {response.text[:300]}...")
            
            if response.status_code == 200:
                print("✅ SUCCESS with token authentication!")
                return True
        
    except Exception as e:
        print(f"❌ Token Error: {e}")
    
    return False

if __name__ == "__main__":
    success = test_hume_correct()
    if success:
        print("\n🎉 Hume AI is functional!")
    else:
        print("\n❌ Hume AI authentication still failing")
        print("💡 Recommendations:")
        print("1. Verify account is active in Hume Portal")
        print("2. Check if API keys were regenerated recently")
        print("3. Ensure account has proper permissions/credits")