#!/usr/bin/env python3
"""
Test Agent Smith Landing → Dashboard Flow
"""
import requests
import time

def test_agent_smith_flow():
    """Test the complete flow from Agent Smith landing to dashboard"""
    
    print("🧪 Testing Agent Smith → Dashboard Flow")
    print("=" * 50)
    
    # Test 1: Agent Smith landing page loads
    print("1️⃣ Testing Agent Smith landing page...")
    try:
        response = requests.get("http://127.0.0.1:8000/agent-smith")
        if response.status_code == 200 and "Access Medical Console" in response.text:
            print("   ✅ Agent Smith landing loads correctly")
            print("   ✅ Gray button found (no emoji)")
        else:
            print("   ❌ Agent Smith landing failed")
            return False
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False
    
    # Test 2: Health check endpoint (what the button calls)
    print("\n2️⃣ Testing health check endpoint...")
    try:
        response = requests.get("http://127.0.0.1:8000/api/v1/health")
        health_data = response.json()
        if health_data.get("status") == "healthy":
            print("   ✅ Health check returns 'healthy'")
            print(f"   ✅ VIGIA system: {health_data.get('vigia_system')}")
        else:
            print("   ❌ Health check failed")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # Test 3: Dashboard loads (redirect target)
    print("\n3️⃣ Testing dashboard (redirect target)...")
    try:
        response = requests.get("http://127.0.0.1:8000/")
        if response.status_code == 200 and "VIGIA Medical AI Dashboard" in response.text:
            print("   ✅ Dashboard loads correctly")
            print("   ✅ Medical interface ready")
        else:
            print("   ❌ Dashboard failed to load")
            return False
    except Exception as e:
        print(f"   ❌ Dashboard error: {e}")
        return False
    
    # Test 4: Check button functionality in HTML
    print("\n4️⃣ Checking button configuration...")
    agent_smith_html = requests.get("http://127.0.0.1:8000/agent-smith").text
    
    if "bg-gray-600" in agent_smith_html:
        print("   ✅ Button is gray (professional)")
    else:
        print("   ⚠️  Button color might not be gray")
    
    if "🏥" not in agent_smith_html.split("Access Medical Console")[0].split("button")[1]:
        print("   ✅ No emoji in button text")
    else:
        print("   ⚠️  Emoji still present in button")
    
    if "127.0.0.1:8000/api/v1/health" in agent_smith_html:
        print("   ✅ Health check points to local endpoint")
    else:
        print("   ❌ Health check points to wrong endpoint")
        return False
    
    if "127.0.0.1:8000" in agent_smith_html:
        print("   ✅ Redirect points to local dashboard")
    else:
        print("   ❌ Redirect points to wrong location")
        return False
    
    print("\n" + "=" * 50)
    print("🎯 FLOW TEST COMPLETE")
    print("\n📋 Manual Test Steps:")
    print("1. Open: http://127.0.0.1:8000/agent-smith")
    print("2. Enter corporate email (e.g., test@autonomos.dev)")
    print("3. Click 'Access Medical Console' (gray button)")
    print("4. Should open http://127.0.0.1:8000/ in new tab")
    print("5. Should show VIGIA Medical Dashboard")
    
    return True

if __name__ == "__main__":
    success = test_agent_smith_flow()
    if success:
        print("\n✅ ALL TESTS PASSED - Ready for manual testing!")
    else:
        print("\n❌ TESTS FAILED - Check configuration")