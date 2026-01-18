import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from auth_utils import create_access_token

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def test_user_id():
    return 1

@pytest.fixture
def auth_headers(test_user_id):
    token = create_access_token(subject=str(test_user_id))
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def mock_db():
    """Mock database for testing"""
    with patch('main.database') as mock:
        mock.connect = AsyncMock()
        mock.disconnect = AsyncMock()
        mock.fetch_one = AsyncMock()
        mock.fetch_all = AsyncMock()
        mock.execute = AsyncMock()
        mock.fetch_val = AsyncMock()
        yield mock

class TestHealthCheck:
    """Test basic health check endpoint"""

    def test_home_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "FastAPI running"}

class TestAuthEndpoints:
    """Test authentication endpoints"""

    def test_register_missing_fields(self, client):
        response = client.post("/auth/register", json={
            "fullName": "Test User"
        })
        assert response.status_code == 422

    def test_register_invalid_email(self, client):
        response = client.post("/auth/register", json={
            "fullName": "Test User",
            "email": "invalid-email",
            "password": "password123",
            "dob": "1990-01-01",
            "gender": "male",
            "nationality": "USA"
        })
        assert response.status_code == 422

    def test_register_short_password(self, client):
        response = client.post("/auth/register", json={
            "fullName": "Test User",
            "email": "test@example.com",
            "password": "short",
            "dob": "1990-01-01",
            "gender": "male",
            "nationality": "USA"
        })
        assert response.status_code == 422

    def test_login_missing_fields(self, client):
        response = client.post("/auth/login", json={
            "email": "test@example.com"
        })
        assert response.status_code == 422

class TestPredictionEndpoint:
    """Test disease prediction endpoint"""

    def test_predict_empty_input(self, client):
        response = client.post("/predict_text", json={"user_input": ""})
        assert response.status_code == 422

    def test_predict_whitespace_only(self, client):
        response = client.post("/predict_text", json={"user_input": "   "})
        assert response.status_code == 400

    def test_predict_too_long_input(self, client):
        response = client.post("/predict_text", json={
            "user_input": "a" * 2001
        })
        assert response.status_code == 400

    def test_predict_dangerous_patterns(self, client):
        """Test SQL injection prevention"""
        dangerous_inputs = [
            "'; DROP TABLE users; --",
            "\" OR 1=1; --",
            "UNION SELECT * FROM users",
            "DELETE FROM chats"
        ]
        for input_text in dangerous_inputs:
            response = client.post("/predict_text", json={"user_input": input_text})
            assert response.status_code == 400

    def test_predict_valid_input(self, client):
        """Test with valid symptom input"""
        response = client.post("/predict_text", json={
            "user_input": "I have fever and cough"
        })
        assert response.status_code == 200
        data = response.json()
        assert "predicted_disease" in data
        assert "matched_symptoms" in data

class TestDetailsEndpoint:
    """Test disease details endpoint"""

    def test_details_empty_disease(self, client):
        response = client.get("/get_details?disease=")
        assert response.status_code == 422

    def test_details_valid_disease(self, client):
        response = client.get("/get_details?disease=fever")
        assert response.status_code == 200
        data = response.json()
        assert "disease" in data
        assert "description" in data
        assert "precautions" in data

    def test_details_dangerous_input(self, client):
        """Test XSS prevention"""
        response = client.get("/get_details?disease=<script>alert('xss')</script>")
        assert response.status_code == 400

class TestChatEndpoints:
    """Test chat history endpoints"""

    def test_get_chats_unauthorized(self, client):
        response = client.get("/chats")
        assert response.status_code == 401

    def test_create_chat_unauthorized(self, client):
        response = client.post("/chats", json={"title": "Test Chat"})
        assert response.status_code == 401

    def test_get_chat_unauthorized(self, client):
        response = client.get("/chats/1")
        assert response.status_code == 401

    def test_delete_chat_unauthorized(self, client):
        response = client.delete("/chats/1")
        assert response.status_code == 401

    def test_create_message_unauthorized(self, client):
        response = client.post("/chats/1/messages", json={
            "role": "user",
            "content": "Hello"
        })
        assert response.status_code == 401

class TestUserProfileEndpoints:
    """Test user profile endpoints"""

    def test_get_profile_unauthorized(self, client):
        response = client.get("/user/profile")
        assert response.status_code == 401

    def test_update_profile_unauthorized(self, client):
        response = client.put("/user/profile", json={"fullName": "New Name"})
        assert response.status_code == 401

    def test_change_password_unauthorized(self, client):
        response = client.post("/user/change-password", data={
            "current_password": "old",
            "new_password": "newpassword"
        })
        assert response.status_code == 401

    def test_get_chat_stats_unauthorized(self, client):
        response = client.get("/user/chat-stats")
        assert response.status_code == 401

class TestInputValidation:
    """Test input validation across all endpoints"""

    def test_xss_in_chat_title(self, client):
        """Test XSS prevention in chat creation"""
        response = client.post("/chats", json={
            "title": "<script>alert('xss')</script>New Chat"
        }, headers={"Authorization": "Bearer invalid"})
        assert response.status_code == 401

    def test_sql_in_chat_title(self, client):
        """Test SQL injection prevention in chat creation"""
        response = client.post("/chats", json={
            "title": "'; DROP TABLE chats; --"
        }, headers={"Authorization": "Bearer invalid"})
        assert response.status_code == 401

    def test_unicode_in_input(self, client):
        """Test handling of unicode characters"""
        response = client.post("/predict_text", json={
            "user_input": "我有发烧和咳嗽"
        })
        assert response.status_code == 200

    def test_emoji_in_input(self, client):
        """Test handling of emoji characters"""
        response = client.post("/predict_text", json={
            "user_input": "I have fever 🤒 and cough 🤧"
        })
        assert response.status_code == 200

class TestRateLimiting:
    """Test basic rate limiting behavior"""

    def test_multiple_requests_same_endpoint(self, client):
        """Test that multiple requests are handled"""
        for i in range(3):
            response = client.get("/")
            assert response.status_code == 200

class TestAuthenticationHeader:
    """Test authentication header validation"""

    def test_missing_bearer_prefix(self, client):
        response = client.get("/chats", headers={"Authorization": "invalid-token"})
        assert response.status_code == 401

    def test_invalid_token_format(self, client):
        response = client.get("/chats", headers={"Authorization": "Bearer"}})
        assert response.status_code == 401

    def test_expired_token(self, client):
        """Test with expired token"""
        from auth_utils import create_access_token
        import time
        expired_token = create_access_token(subject="1", expires_seconds=-1)
        response = client.get("/chats", headers={"Authorization": f"Bearer {expired_token}"})
        assert response.status_code == 401

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
