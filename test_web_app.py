#!/usr/bin/env python3
"""
Quick test for web app functionality
"""

import requests
import sys
from pathlib import Path

def test_web_app():
    """Test if the web app is running and responsive"""
    
    print("🧪 Testing Baby Cry Classification Web App")
    print("=" * 50)
    
    # Test if the server is running
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        if response.status_code == 200:
            print("✅ Web app is running successfully!")
            print(f"   Status: {response.status_code}")
            print(f"   URL: http://localhost:5000")
            
            # Check if it contains expected content
            if "Baby Cry Classification" in response.text:
                print("✅ App content loaded correctly")
            else:
                print("⚠️  App content may have issues")
            
            print("\n🎯 Ready for testing!")
            print("   1. Upload files from test_dataset/")
            print("   2. Try different cry types")
            print("   3. Check classification results")
            
            return True
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to web app!")
        print("   Make sure to run: python app.py")
        return False
    except requests.exceptions.Timeout:
        print("❌ Web app not responding (timeout)")
        return False
    except Exception as e:
        print(f"❌ Error testing web app: {e}")
        return False

if __name__ == "__main__":
    success = test_web_app()
    
    if not success:
        print("\n💡 To start the web app:")
        print("   cd d:\\Baby-Cry-Model")
        print("   python web_app/app.py")
        sys.exit(1)
    else:
        print("\n🎉 Web app test passed! Ready for demonstration.")