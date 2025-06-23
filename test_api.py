import json
import requests

# API URL - change as needed
BASE_URL = "http://localhost:8000"

def print_json(data):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2))

def test_api():
    """Test the NLP to SQL API"""
    print("Testing NLP to SQL API...")

    # Check if API is running
    print("\n1. Checking API status...")
    response = requests.get(f"{BASE_URL}/")
    print_json(response.json())

    # Create a new session
    print("\n2. Creating a new session...")
    session_data = {
        "db_name": "Adventureworks",  # Change to match your database
        "username": "postgres",
        "password": "akshwalia", 
        "use_memory": True,
        "use_cache": True
    }
    response = requests.post(f"{BASE_URL}/sessions", json=session_data)
    
    if response.status_code == 201:
        session_result = response.json()
        print_json(session_result)
        session_id = session_result["session_id"]
    else:
        print(f"Failed to create session: {response.status_code}")
        print_json(response.json())
        return

    # Get session info
    print(f"\n3. Getting session info for {session_id}...")
    response = requests.get(f"{BASE_URL}/sessions/{session_id}")
    print_json(response.json())

    # Make a query within session
    print("\n4. Making a query with session...")
    query_data = {
        "question": "Show me the last 5 sales",
        "auto_fix": True
    }
    response = requests.post(f"{BASE_URL}/sessions/{session_id}/query", json=query_data)
    print_json(response.json())

    # Make a follow-up query to test memory
    print("\n5. Making a follow-up query to test memory...")
    query_data = {
        "question": "Which of those had the highest subtotal?",
        "auto_fix": True
    }
    response = requests.post(f"{BASE_URL}/sessions/{session_id}/query", json=query_data)
    print_json(response.json())

    # Make a query without session
    print("\n6. Making a query without session...")
    query_data = {
        "question": "Show me the total number of customers"
    }
    response = requests.post(f"{BASE_URL}/query", json=query_data)
    print_json(response.json())

    # List all sessions
    print("\n7. Listing all sessions...")
    response = requests.get(f"{BASE_URL}/sessions")
    print_json(response.json())

    # Delete the session
    print(f"\n8. Deleting session {session_id}...")
    response = requests.delete(f"{BASE_URL}/sessions/{session_id}")
    print_json(response.json())

if __name__ == "__main__":
    test_api() 