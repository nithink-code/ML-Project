"""
Test script to verify backend connectivity and configuration
"""
import requests
import json

BACKEND_URL = "http://localhost:8000"

def test_backend_connection():
    """Test if backend is running and responding"""
    print("Testing backend connection...")
    try:
        # Test health endpoint (if exists)
        response = requests.get(f"{BACKEND_URL}/api/auth/dev-login", timeout=5)
        print(f"✓ Backend is reachable (Status: {response.status_code})")
        return True
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to backend. Is the server running?")
        print("  Try running: cd backend && python server.py")
        return False
    except Exception as e:
        print(f"✗ Error connecting to backend: {e}")
        return False

def test_cors_headers():
    """Test CORS headers"""
    print("\nTesting CORS configuration...")
    try:
        response = requests.options(
            f"{BACKEND_URL}/api/interviews",
            headers={
                'Origin': 'http://localhost:3000',
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'content-type'
            },
            timeout=5
        )
        
        cors_origin = response.headers.get('Access-Control-Allow-Origin')
        if cors_origin:
            print(f"✓ CORS configured: {cors_origin}")
        else:
            print("✗ CORS not properly configured")
            print(f"  Response headers: {dict(response.headers)}")
        
        return cors_origin is not None
    except Exception as e:
        print(f"✗ Error testing CORS: {e}")
        return False

def main():
    print("=" * 60)
    print("Backend Connection Test")
    print("=" * 60)
    
    backend_ok = test_backend_connection()
    cors_ok = test_cors_headers()
    
    print("\n" + "=" * 60)
    if backend_ok and cors_ok:
        print("✓ All tests passed! Backend is ready.")
    else:
        print("✗ Some tests failed. Please check the issues above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
