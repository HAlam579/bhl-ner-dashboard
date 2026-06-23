"""
Debug script to inspect the BHL API response structure.
Run with: py tests/debug_api.py
"""

import sys
import os
import json
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.fetch_bhl import API_KEY, BASE_URL


def debug_item_metadata(item_id):
    """Print the full API response to understand the structure."""
    
    print("=" * 60)
    print(f"Debugging GetItemMetadata for Item ID: {item_id}")
    print("=" * 60)
    
    params = {
        "op": "GetItemMetadata",
        "id": item_id,
        "apikey": API_KEY,
        "format": "json"
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response keys: {list(data.keys())}")
            print(f"Status: {data.get('Status')}")
            
            result = data.get("Result")
            print(f"Result type: {type(result)}")
            
            if isinstance(result, dict):
                print(f"Result keys: {list(result.keys())}")
                
                # Check for Pages
                if "Pages" in result:
                    pages = result.get("Pages")
                    print(f"Pages type: {type(pages)}")
                    if pages:
                        print(f"Number of pages: {len(pages)}")
                        if len(pages) > 0:
                            print(f"First page keys: {list(pages[0].keys())}")
                    else:
                        print("Pages is empty or None")
                else:
                    print("No 'Pages' field found in Result")
                    
                    # Try other fields that might contain page info
                    if "Items" in result:
                        print(f"Items found: {len(result.get('Items', []))}")
                    if "Parts" in result:
                        print(f"Parts found: {len(result.get('Parts', []))}")
                        
            elif isinstance(result, list):
                print(f"Result is a list with {len(result)} items")
                if result:
                    print(f"First item keys: {list(result[0].keys()) if isinstance(result[0], dict) else 'Not a dict'}")
            else:
                print(f"Unexpected result: {result}")
                
            print("\nFull response (first 1000 characters):")
            print("-" * 40)
            print(json.dumps(data, indent=2)[:1000])
                
        else:
            print(f"Error: {response.status_code}")
            print(response.text[:500])
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Test with the item we found
    item_id = 240152
    debug_item_metadata(item_id)