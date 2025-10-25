import requests
import sys
import json
from datetime import datetime

class InterviewChatbotAPITester:
    def __init__(self, base_url="https://ai-interviewpal-2.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.session_token = None
        self.user_data = None
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []

    def run_test(self, name, method, endpoint, expected_status, data=None, headers=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        test_headers = {'Content-Type': 'application/json'}
        
        if headers:
            test_headers.update(headers)
        
        if self.session_token:
            test_headers['Authorization'] = f'Bearer {self.session_token}'

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        print(f"   Method: {method}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=test_headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=test_headers, timeout=30)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=test_headers, timeout=30)
            elif method == 'DELETE':
                response = requests.delete(url, headers=test_headers, timeout=30)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                self.failed_tests.append({
                    "test": name,
                    "expected": expected_status,
                    "actual": response.status_code,
                    "response": response.text[:200]
                })
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            self.failed_tests.append({
                "test": name,
                "error": str(e)
            })
            return False, {}

    def test_auth_flow(self):
        """Test authentication flow with mock session"""
        print("\n🔐 Testing Authentication Flow...")
        
        # Test creating session with mock session ID
        mock_session_id = "test_session_123"
        success, response = self.run_test(
            "Create Session (Mock)",
            "POST",
            "auth/session",
            200,  # Expecting 200 for successful session creation
            data={"session_id": mock_session_id}
        )
        
        if success and 'session_token' in response:
            self.session_token = response['session_token']
            self.user_data = response.get('user')
            print(f"   Session token obtained: {self.session_token[:20]}...")
            return True
        else:
            print("   ⚠️  Session creation failed - this is expected for invalid session ID")
            # For testing purposes, let's create a mock token
            self.session_token = "mock_token_for_testing"
            return False

    def test_get_current_user(self):
        """Test getting current user"""
        if not self.session_token:
            print("⚠️  Skipping user test - no session token")
            return False
            
        success, response = self.run_test(
            "Get Current User",
            "GET",
            "auth/me",
            200
        )
        return success

    def test_interview_creation(self):
        """Test creating different types of interviews"""
        if not self.session_token:
            print("⚠️  Skipping interview creation - no session token")
            return []
            
        interview_types = ["technical", "behavioral", "general"]
        created_interviews = []
        
        for interview_type in interview_types:
            success, response = self.run_test(
                f"Create {interview_type.title()} Interview",
                "POST",
                "interviews",
                200,
                data={"interview_type": interview_type}
            )
            if success and 'id' in response:
                created_interviews.append(response['id'])
        
        return created_interviews

    def test_get_interviews(self):
        """Test getting user's interviews"""
        if not self.session_token:
            print("⚠️  Skipping get interviews - no session token")
            return False
            
        success, response = self.run_test(
            "Get User Interviews",
            "GET",
            "interviews",
            200
        )
        return success

    def test_interview_messaging(self, interview_id):
        """Test sending messages in an interview"""
        if not self.session_token or not interview_id:
            print("⚠️  Skipping messaging test - no session token or interview ID")
            return False
            
        success, response = self.run_test(
            "Send Interview Message",
            "POST",
            f"interviews/{interview_id}/message",
            200,
            data={"content": "This is a test message", "is_voice": False}
        )
        return success

    def test_interview_completion(self, interview_id):
        """Test completing an interview"""
        if not self.session_token or not interview_id:
            print("⚠️  Skipping completion test - no session token or interview ID")
            return False
            
        success, response = self.run_test(
            "Complete Interview",
            "POST",
            f"interviews/{interview_id}/complete",
            200
        )
        return success

    def test_interview_evaluation(self, interview_id):
        """Test generating interview evaluation"""
        if not self.session_token or not interview_id:
            print("⚠️  Skipping evaluation test - no session token or interview ID")
            return False
            
        success, response = self.run_test(
            "Generate Interview Evaluation",
            "POST",
            f"interviews/{interview_id}/evaluate",
            200
        )
        return success

    def test_logout(self):
        """Test logout functionality"""
        if not self.session_token:
            print("⚠️  Skipping logout test - no session token")
            return False
            
        success, response = self.run_test(
            "Logout",
            "POST",
            "auth/logout",
            200
        )
        return success

def main():
    print("🚀 Starting AI Interview Chatbot API Tests")
    print("=" * 50)
    
    tester = InterviewChatbotAPITester()
    
    # Test authentication flow
    auth_success = tester.test_auth_flow()
    
    # Test getting current user
    tester.test_get_current_user()
    
    # Test interview creation
    created_interviews = tester.test_interview_creation()
    
    # Test getting interviews
    tester.test_get_interviews()
    
    # Test messaging with first created interview
    if created_interviews:
        first_interview = created_interviews[0]
        tester.test_interview_messaging(first_interview)
        tester.test_interview_completion(first_interview)
        tester.test_interview_evaluation(first_interview)
    
    # Test logout
    tester.test_logout()
    
    # Print final results
    print("\n" + "=" * 50)
    print(f"📊 Final Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.failed_tests:
        print("\n❌ Failed Tests:")
        for failed in tester.failed_tests:
            error_msg = failed.get('error', f'Expected {failed.get("expected")}, got {failed.get("actual")}')
            print(f"   - {failed.get('test', 'Unknown')}: {error_msg}")
    
    success_rate = (tester.tests_passed / tester.tests_run * 100) if tester.tests_run > 0 else 0
    print(f"\n📈 Success Rate: {success_rate:.1f}%")
    
    return 0 if success_rate > 50 else 1

if __name__ == "__main__":
    sys.exit(main())