"""
Example script to test the API Gateway
Run after starting services with: docker compose up api-gateway sentiment-service postgres
"""
import requests
import json
from datetime import datetime

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_analyze_single():
    """Test single segment analysis"""
    print("🔍 Testing single segment analysis...")
    
    segment_data = {
        "segment_id": "test_123",
        "channel_title": "France 2",
        "channel_name": "france2",
        "segment_start": "2024-01-15T20:00:00",
        "segment_end": "2024-01-15T20:05:00",
        "duration_seconds": "300",
        "report_text": "Un excellent reportage sur l'agriculture biologique en France. Les agriculteurs témoignent de leur transition vers des pratiques durables.",
        "llm_category": "agriculture_alimentation"
    }
    
    response = requests.post(f"{API_URL}/analyze", json=segment_data)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()
    
    return response.json().get("segment_id")

def test_batch_analysis():
    """Test batch analysis"""
    print("📦 Testing batch analysis...")
    
    batch_data = {
        "segments": [
            {
                "segment_id": "batch_001",
                "channel_title": "France 2",
                "channel_name": "france2",
                "segment_start": "2024-01-15T20:00:00",
                "segment_end": "2024-01-15T20:05:00",
                "duration_seconds": "300",
                "report_text": "Un reportage positif sur les énergies renouvelables en France.",
                "llm_category": "energy"
            },
            {
                "segment_id": "batch_002",
                "channel_title": "TF1",
                "channel_name": "tf1",
                "segment_start": "2024-01-15T21:00:00",
                "segment_end": "2024-01-15T21:05:00",
                "duration_seconds": "300",
                "report_text": "La pollution des transports continue d'augmenter dans les grandes villes.",
                "llm_category": "mobility_transport"
            },
            {
                "segment_id": "batch_003",
                "channel_title": "M6",
                "channel_name": "m6",
                "segment_start": "2024-01-15T22:00:00",
                "segment_end": "2024-01-15T22:05:00",
                "duration_seconds": "300",
                "report_text": "Les agriculteurs manifestent contre les nouvelles réglementations.",
                "llm_category": "agriculture_alimentation"
            }
        ]
    }
    
    response = requests.post(f"{API_URL}/analyze/batch", json=batch_data)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_get_results(segment_id):
    """Test retrieving results by segment ID"""
    print(f"📊 Testing get results for segment {segment_id}...")
    
    response = requests.get(f"{API_URL}/results/{segment_id}")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(response.text)
    print()

def main():
    """Run all tests"""
    print("=" * 60)
    print("API Gateway Test Suite")
    print("=" * 60)
    print()
    
    try:
        # Test health
        test_health()
        
        # Test single analysis
        segment_id = test_analyze_single()
        
        # Test batch analysis
        test_batch_analysis()
        
        # Test getting results
        if segment_id:
            test_get_results(segment_id)
        
        print("✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API. Make sure services are running:")
        print("   docker compose up api-gateway sentiment-service postgres")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
