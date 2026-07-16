import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BHL_API_KEY")

if not API_KEY:
    raise ValueError("BHL_API_KEY not found in environment variables. Please set it in your .env file.")


BASE_URL = "https://www.biodiversitylibrary.org/api3"


def test_api_connection():
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
    Args:
        page_id (int): The ID of the page to fetch
        
    Returns:
        str: The text content, or None if error
    """
    params = {
        "op": "GetPageMetadata",
        "pageid": page_id,
        "ocr": "true",
        "apikey": API_KEY,
        "format": "json"
    }
    
    try:
        print(f"Fetching page {page_id}...")
        response = requests.get(BASE_URL, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("Status") == "ok":
                result = data.get("Result", {})
                
                # Handle both dict and list responses
                if isinstance(result, list):
                    print(f"  Result is a list with {len(result)} items")
                    if result:
                        result = result[0]  # Take the first item
                    else:
                        print("  Result list is empty")
                        return None
                
                # Now result should be a dict
                if isinstance(result, dict):
                    # The OCR text could be in different fields
                    # Try Page.OcrText first
                    page = result.get("Page", {})
                    if isinstance(page, dict):
                        text = page.get("OcrText")
                        if text:
                            return text
                    
                    # Try direct OcrText field
                    text = result.get("OcrText")
                    if text:
                        return text
                    
                    # Try Text field
                    text = result.get("Text")
                    if text:
                        return text
                    
                    print(f"Warning: No text found for page {page_id}")
                    print(f"  Available keys: {list(result.keys())}")
                    return None
                else:
                    print(f"Warning: Unexpected Result format: {type(result)}")
                    return None
            else:
                print(f"API Error: {data.get('ErrorMessage', 'Unknown')}")
                return None
        else:
            print(f"HTTP Error: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"Error: Request timed out for page {page_id}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"Error: Connection error for page {page_id}")
        return None
    except Exception as e:
        print(f"Error: Unexpected error fetching page {page_id}: {e}")
        return None

def search_publications(search_term, limit=5, retries=2):
    """    
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
                result = data.get("Result")
                
                # Handle both cases: Result can be a dict or a list
                if isinstance(result, dict):
                    return result
                elif isinstance(result, list) and result:
                    # If it's a list, return the first item
                    return result[0]
                else:
                    print(f"Unexpected Result format: {type(result)}")
                    return None
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
    Args:
        search_term (str): Search term
        
    Returns:
        int: Page ID or None
    """
    results = search_publications(search_term, limit=1)
    if not results:
        return None
    
    item = results[0]
    
    # Method 1: Get ThumbnailPageID directly from search results
    page_id = item.get("ThumbnailPageID")
    if page_id:
        print(f"Found first page ID from search: {page_id}")
        return page_id
    
    # Method 2: If not in search results, try item metadata
    item_id = item.get("ItemID")
    if not item_id:
        print("No ItemID found in search results.")
        return None
    
    metadata = get_item_metadata(item_id)
    if metadata:
        # If metadata is a list, get the first item
        if isinstance(metadata, list) and metadata:
            metadata = metadata[0]
        
        # Try to get ThumbnailPageID from metadata
        page_id = metadata.get("ThumbnailPageID")
        if page_id:
            print(f"Found first page ID from metadata: {page_id}")
            return page_id
    
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