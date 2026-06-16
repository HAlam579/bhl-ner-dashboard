"""
Module for fetching text from the Biodiversity Heritage Library API.
Includes comprehensive error handling.
"""

import requests
import json
import time

API_KEY = "e68afbdd-1f4b-4915-9758-14861dcdd407"
BASE_URL = "https://www.biodiversitylibrary.org/api3"


def test_api_connection():
    """
    Test if the API key works and we can fetch data.
    Includes error handling for network issues.
    """
    print("Testing BHL API connection...")
    print(f"API Key: {API_KEY[:10]}... (hidden for security)")
    print()
    
    params = {
        "op": "PublicationSearch",
        "searchterm": "butterfly",
        "apikey": API_KEY,
        "format": "json"
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print("✓ API connection successful!")
                print(f"  Response contains: {list(data.keys())}")
                
                if data.get("Status") == "ok":
                    results = data.get("Result", [])
                    print(f"  Number of results: {len(results)}")
                    if len(results) > 0:
                        first_result = results[0]
                        print(f"  First result title: {first_result.get('Title', 'No title')}")
                else:
                    print(f"  API error: {data.get('ErrorMessage', 'Unknown')}")
            except json.JSONDecodeError as e:
                print(f"✗ Error: The API returned invalid JSON: {e}")
                print(f"  Response starts with: {response.text[:100]}")
        else:
            print(f"✗ HTTP Error: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            
    except requests.exceptions.Timeout:
        print("✗ Error: The API request timed out.")
        print("  The server may be slow. Try again later.")
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to the BHL API.")
        print("  Please check your internet connection.")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


def get_page_text(page_id):
    """
    Fetch the text content of a specific page with robust error handling.
    
    Args:
        page_id (int): The ID of the page to fetch
        
    Returns:
        str: The text content, or None if error
    """
    params = {
        "op": "GetPageText",
        "id": page_id,
        "apikey": API_KEY,
        "format": "json"
    }
    
    try:
        print(f"Fetching page {page_id}...")
        response = requests.get(BASE_URL, params=params, timeout=30)
        
        if response.status_code == 200:
            try:
                data = response.json()
                if data.get("Status") == "ok":
                    result = data.get("Result", {})
                    
                    # Try to get text from various fields
                    if isinstance(result, dict):
                        text = result.get("Text")
                        if text:
                            return text
                        text = result.get("PageText")
                        if text:
                            return text
                    
                    # If no text found, return None
                    print(f"Warning: No text found for page {page_id}")
                    return None
                else:
                    print(f"API Error: {data.get('ErrorMessage', 'Unknown')}")
                    return None
            except json.JSONDecodeError as e:
                print(f"Error: The API returned invalid JSON: {e}")
                return None
        else:
            print(f"HTTP Error: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"Error: Request timed out for page {page_id}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"Error: Connection error for page {page_id}")
        print("  Please check your internet connection.")
        return None
    except Exception as e:
        print(f"Error: Unexpected error fetching page {page_id}: {e}")
        return None


def search_publications(search_term, limit=5, retries=2):
    """
    Search for publications by keyword with retry logic.
    
    Args:
        search_term (str): Keyword to search for
        limit (int): Maximum number of results
        retries (int): Number of times to retry on failure
        
    Returns:
        list: List of publication results, or empty list if error
    """
    params = {
        "op": "PublicationSearch",
        "searchterm": search_term,
        "apikey": API_KEY,
        "format": "json"
    }
    
    for attempt in range(retries + 1):
        try:
            print(f"Searching for '{search_term}' (attempt {attempt + 1})...")
            response = requests.get(BASE_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("Status") == "ok":
                    results = data.get("Result", [])[:limit]
                    print(f"  Found {len(results)} results")
                    return results
                else:
                    print(f"  API Error: {data.get('ErrorMessage', 'Unknown')}")
                    if attempt < retries:
                        print(f"  Retrying...")
                        time.sleep(2)  # Wait 2 seconds before retry
                    else:
                        return []
            else:
                print(f"  HTTP Error: {response.status_code}")
                if attempt < retries:
                    print(f"  Retrying...")
                    time.sleep(2)
                else:
                    return []
                    
        except requests.exceptions.Timeout:
            print(f"  Timeout error (attempt {attempt + 1})")
            if attempt < retries:
                print(f"  Retrying...")
                time.sleep(3)
            else:
                return []
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < retries:
                print(f"  Retrying...")
                time.sleep(2)
            else:
                return []
    
    return []


def get_item_metadata(item_id):
    """
    Get metadata for an item with error handling.
    
    Args:
        item_id (int): Item ID
        
    Returns:
        dict: Item metadata, or None if error
    """
    params = {
        "op": "GetItemMetadata",
        "id": item_id,
        "apikey": API_KEY,
        "format": "json"
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get("Status") == "ok":
                return data.get("Result", {})
            else:
                print(f"API Error: {data.get('ErrorMessage', 'Unknown')}")
                return None
        else:
            print(f"HTTP Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_first_page_id_from_search(search_term):
    """
    Search for a term and get the first page ID with error handling.
    
    Args:
        search_term (str): Search term
        
    Returns:
        int: Page ID or None
    """
    results = search_publications(search_term, limit=1)
    if not results:
        return None
    
    item_id = results[0].get("ItemID")
    if not item_id:
        print("No ItemID found in search results.")
        return None
    
    metadata = get_item_metadata(item_id)
    if not metadata:
        return None
    
    pages = metadata.get("Pages", [])
    if not pages:
        print("No pages found for this item.")
        return None
    
    first_page = pages[0]
    page_id = first_page.get("PageID")
    if page_id:
        print(f"Found first page ID: {page_id}")
        return page_id
    else:
        print("No PageID found.")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("BHL API Test (with Error Handling)")
    print("=" * 60)
    
    # Test 1: API connection
    print("\n--- Test 1: API Connection ---")
    test_api_connection()
    
    # Test 2: Search for something
    print("\n--- Test 2: Search Publications ---")
    results = search_publications("butterfly", limit=2)
    if results:
        for item in results:
            print(f"  - {item.get('Title', 'No title')}")
    
    # Test 3: Try to find a page
    print("\n--- Test 3: Find First Page ---")
    page_id = get_first_page_id_from_search("butterfly")
    if page_id:
        print(f"  PageID found: {page_id}")
    else:
        print("  No PageID found.")
    
    print("\n" + "=" * 60)
    print("Test complete.")
    print("=" * 60)