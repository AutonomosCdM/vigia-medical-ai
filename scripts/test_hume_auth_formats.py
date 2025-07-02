#!/usr/bin/env python3
"""
Test different Hume AI authentication formats
"""

import asyncio
import os
import aiohttp
import base64
from pathlib import Path
from dotenv import load_dotenv

# Load environment
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

async def test_api_key_auth():
    """Test with X-Hume-Api-Key header"""
    api_key = os.getenv("HUME_AI_API_KEY")
    
    async with aiohttp.ClientSession() as session:
        headers = {'X-Hume-Api-Key': api_key}
        
        async with session.get("https://api.hume.ai/v0/batch/jobs", headers=headers) as response:
            print(f"API Key Auth - Status: {response.status}")
            text = await response.text()
            print(f"Response: {text[:200]}")
            return response.status == 200

async def test_basic_auth():
    """Test with Basic authentication using API key + secret"""
    api_key = os.getenv("HUME_AI_API_KEY")
    secret_key = os.getenv("HUME_AI_SECRET_KEY")
    
    if not secret_key:
        print("No secret key found")
        return False
    
    # Create Basic auth header
    credentials = f"{api_key}:{secret_key}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    
    async with aiohttp.ClientSession() as session:
        headers = {'Authorization': f'Basic {encoded_credentials}'}
        
        async with session.get("https://api.hume.ai/v0/batch/jobs", headers=headers) as response:
            print(f"Basic Auth - Status: {response.status}")
            text = await response.text()
            print(f"Response: {text[:200]}")
            return response.status == 200

async def test_bearer_auth():
    """Test with Bearer token using API key"""
    api_key = os.getenv("HUME_AI_API_KEY")
    
    async with aiohttp.ClientSession() as session:
        headers = {'Authorization': f'Bearer {api_key}'}
        
        async with session.get("https://api.hume.ai/v0/batch/jobs", headers=headers) as response:
            print(f"Bearer Auth - Status: {response.status}")
            text = await response.text()
            print(f"Response: {text[:200]}")
            return response.status == 200

async def main():
    """Test all authentication methods"""
    print("🔑 Testing Hume AI Authentication Methods")
    print("=" * 50)
    
    print("\\n1. Testing X-Hume-Api-Key header:")
    success1 = await test_api_key_auth()
    
    print("\\n2. Testing Basic Authentication:")
    success2 = await test_basic_auth()
    
    print("\\n3. Testing Bearer Token:")
    success3 = await test_bearer_auth()
    
    print("\\n" + "=" * 50)
    print(f"X-Hume-Api-Key: {'✅' if success1 else '❌'}")
    print(f"Basic Auth: {'✅' if success2 else '❌'}")
    print(f"Bearer Token: {'✅' if success3 else '❌'}")
    
    return success1 or success2 or success3

if __name__ == "__main__":
    success = asyncio.run(main())
    print(f"\\nResult: {'✅' if success else '❌'}")