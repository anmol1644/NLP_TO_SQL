import os
import requests
import json
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# API endpoint
API_BASE_URL = "http://localhost:8000"

def test_create_session():
    """Test creating a new session"""
    url = f"{API_BASE_URL}/sessions"
    
    # Database connection details
    payload = {
        "db_name": os.getenv("DB_NAME"),
        "username": os.getenv("DB_USERNAME"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "use_memory": True,
        "use_cache": True
    }
    
    response = requests.post(url, json=payload)
    print("\n=== Creating Session ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 201:
        return response.json()["session_id"]
    else:
        return None

def test_sql_query(session_id, query):
    """Test executing a regular SQL query"""
    url = f"{API_BASE_URL}/sessions/{session_id}/query"
    
    # Query details
    payload = {
        "question": query,
        "auto_fix": True,
        "max_attempts": 2
    }
    
    response = requests.post(url, json=payload)
    print(f"\n=== SQL Query: '{query}' ===")
    print(f"Status Code: {response.status_code}")
    
    # Format the results for readability
    result = response.json()
    
    # Print SQL and text response
    print(f"SQL: {result.get('sql')}")
    print(f"Text Response: {result.get('text', 'No text response')}")
    
    # Format and print results if available
    if result.get("results") and len(result["results"]) > 0:
        print(f"Results: (showing {min(3, len(result['results']))} of {len(result['results'])} rows)")
        for i, row in enumerate(result["results"][:3]):
            print(f"  Row {i+1}: {json.dumps(row, indent=2)}")
    else:
        print("No results or empty result set")
    
    return result

def test_conversational_query(session_id, query):
    """Test executing a conversational query that returns only text"""
    url = f"{API_BASE_URL}/sessions/{session_id}/query"
    
    # Query details
    payload = {
        "question": query,
    }
    
    response = requests.post(url, json=payload)
    print(f"\n=== Conversational Query: '{query}' ===")
    print(f"Status Code: {response.status_code}")
    
    # Format the results for readability
    result = response.json()
    
    # Print text response and check if it's a conversational query
    print(f"Is Conversational: {result.get('is_conversational', False)}")
    print(f"Text Response: {result.get('text', 'No text response')}")
    
    return result

def test_analysis_query(session_id, query):
    """Test executing an analysis query that might require multiple queries"""
    url = f"{API_BASE_URL}/sessions/{session_id}/query"
    
    # Query details
    payload = {
        "question": query,
        "auto_fix": True
    }
    
    response = requests.post(url, json=payload)
    print(f"\n=== Analysis Query: '{query}' ===")
    print(f"Status Code: {response.status_code}")
    
    # Format the results for readability
    result = response.json()
    
    # Check query type and analysis type
    query_type = result.get("query_type", "standard")
    analysis_type = result.get("analysis_type", "")
    is_multi_query = result.get("is_multi_query", False)
    
    print(f"Query Type: {query_type}")
    if analysis_type:
        print(f"Analysis Type: {analysis_type}")
    print(f"Is Multi-Query Analysis: {is_multi_query}")
    
    # Check if this is a multi-query or causal (why) analysis
    if (query_type == "analysis" and "tables" in result):
        # Print all tables with headers
        print("\nMultiple tables returned:")
        for i, table in enumerate(result["tables"]):
            print(f"\nTable {i+1}: {table.get('name', 'Unnamed Query')}")
            print(f"Description: {table.get('description', 'No description')}")
            print(f"SQL: {table.get('sql', 'No SQL')}")
            
            # Format results
            if table.get("results") and len(table["results"]) > 0:
                print(f"Results: (showing {min(3, len(table['results']))} of {len(table['results'])} rows)")
                for j, row in enumerate(table["results"][:3]):
                    print(f"  Row {j+1}: {json.dumps(row, indent=2)}")
            else:
                print("  No results or empty result set")
        
        # Print analysis text
        print(f"\nAnalysis:\n{result.get('text', 'No analysis provided')}")
    else:
        # Print single SQL and response
        print(f"SQL: {result.get('sql')}")
        print(f"Text Response: {result.get('text', 'No text response')}")
        
        # Format and print results if available
        if result.get("results") and len(result["results"]) > 0:
            print(f"Results: (showing {min(3, len(result['results']))} of {len(result['results'])} rows)")
            for i, row in enumerate(result["results"][:3]):
                print(f"  Row {i+1}: {json.dumps(row, indent=2)}")
        else:
            print("No results or empty result set")
    
    return result

def test_pagination(session_id, query):
    """Test executing a query that returns paginated results"""
    url = f"{API_BASE_URL}/sessions/{session_id}/query"
    
    # Query details
    payload = {
        "question": query,
        "auto_fix": True
    }
    
    # Execute the initial query
    response = requests.post(url, json=payload)
    print(f"\n=== Pagination Test: '{query}' ===")
    print(f"Status Code: {response.status_code}")
    
    # Format the results for readability
    result = response.json()
    
    # Check if pagination is available
    if "pagination" in result:
        pagination = result["pagination"]
        table_id = pagination["table_id"]
        total_rows = pagination["total_rows"]
        total_pages = pagination["total_pages"]
        
        print(f"Query returned {total_rows} rows with pagination")
        print(f"Table ID: {table_id}")
        print(f"Total pages: {total_pages}")
        print(f"First page results: {len(result['results'])} rows")
        
        # Print first page results
        for i, row in enumerate(result["results"]):
            print(f"  Row {i+1}: {json.dumps(row, indent=2)}")
        
        # If there are more pages, fetch the second page
        if total_pages > 1:
            page_url = f"{API_BASE_URL}/sessions/{session_id}/results/{table_id}?page=2"
            page_response = requests.get(page_url)
            
            if page_response.status_code == 200:
                page_result = page_response.json()
                print(f"\nSecond page results: {len(page_result['results'])} rows")
                
                # Print second page results
                for i, row in enumerate(page_result["results"]):
                    print(f"  Row {i+1}: {json.dumps(row, indent=2)}")
            else:
                print(f"Failed to fetch second page: {page_response.status_code}")
    else:
        print("No pagination available for this query")
        print(f"SQL: {result.get('sql')}")
        print(f"Results: {len(result.get('results', []))} rows")
    
    return result

if __name__ == "__main__":
    # Create a session
    session_id = test_create_session()
    
    if session_id:
        # Test regular SQL query
        test_sql_query(session_id, "Show me the top 5 customers by total order amount")
        
        # Test a conversational query (no SQL)
        test_conversational_query(session_id, "What is a relational database?")
        
        # Test a follow-up conversational query
        test_conversational_query(session_id, "How does it differ from NoSQL?")
        
        # Test LLM-related conversational queries
        test_conversational_query(session_id, "Tell me which LLM are you using behind the scenes")
        test_conversational_query(session_id, "What model powers your responses?")
        test_conversational_query(session_id, "How does this system work?")
        
        # Test a single-query analysis
        test_analysis_query(session_id, "Show me the sales for 2014 and provide a brief analysis")
        
        # Test a comparative analysis query (likely multi-query)
        test_analysis_query(session_id, "Compare sales from 2014 versus 2013 and explain any significant differences")
        
        # Test a "why" analysis question
        test_analysis_query(session_id, "Why are our sales decreasing compared to last quarter of 2013?")
        
        # Test pagination
        test_pagination(session_id, "List all sales from 2014 and 2013")
    else:
        print("Failed to create session. Check your database connection settings.") 