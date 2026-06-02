"""
Module for fetching text from the Biodiversity Heritage Library API.
"""

import requests
import json

API_KEY = "e68afbdd-1f4b-4915-9758-14861dcdd407"
BASE_URL = "https://www.biodiversitylibrary.org/api3"

def test_api_connection():
    """Test if the API key works and we can fetch data."""
    
    print("Testing BHL API connection...")
    print(f"API Key: {API_KEY[:10]}... (hidden for security)")
    print()
    
    # Try a simple search for publications about "butterfly"
    params = {
        "op": "PublicationSearch",
        "searchterm": "butterfly",
        "apikey": API_KEY,
        "format": "json"
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✓ API connection successful!")
            print(f"  Response contains: {list(data.keys())}")
            
            # Try to get results
            if "results" in data:
                print(f"  Number of results: {len(data['results'])}")
                if len(data['results']) > 0:
                    first_result = data['results'][0]
                    print(f"  First result title: {first_result.get('Title', 'No title')}")
            else:
                print(f"  Full response: {json.dumps(data, indent=2)[:500]}")
        else:
            print(f"✗ API error: {response.text}")
            
    except Exception as e:
        print(f"✗ Exception occurred: {e}")

def get_page_text(page_id):
    """
    Fetch the text content of a specific page.
    
    Args:
        page_id (int): The ID of the page to fetch
        
    Returns:
        dict or None: The API response or None if error
    """
    params = {
        "op": "GetPageText",
        "id": page_id,
        "apikey": API_KEY,
        "format": "json"
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching page {page_id}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

if __name__ == "__main__":
    test_api_connection()