"""
Test script for BHL API v3 GetPageMetadata fix.
Run with: py tests/test_api_fix.py
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.fetch_bhl import get_page_text, search_publications, get_item_metadata


def test_api_fix():
    """Test the corrected BHL API call."""
    
    print("=" * 60)
    print("Testing BHL API v3 GetPageMetadata")
    print("=" * 60)
    
    # 1. Search for a publication
    print("\n1. Searching for 'butterfly'...")
    results = search_publications("butterfly", limit=1)
    
    if not results:
        print("❌ No results found. Check your API key.")
        return
    
    item = results[0]
    item_id = item.get("ItemID")
    title = item.get("Title", "Unknown title")
    print(f"   ✓ Found: {title}")
    print(f"   Item ID: {item_id}")
    
    # 2. Get ThumbnailPageID from search results
    print(f"\n2. Getting PageID from search results...")
    page_id = item.get("ThumbnailPageID")
    
    if page_id:
        print(f"   ✓ Found PageID: {page_id}")
    else:
        print("   ❌ No ThumbnailPageID in search results.")
        # Try item metadata as fallback
        print(f"\n3. Trying item metadata as fallback...")
        metadata = get_item_metadata(item_id)
        if metadata:
            if isinstance(metadata, list) and metadata:
                metadata = metadata[0]
            page_id = metadata.get("ThumbnailPageID")
            if page_id:
                print(f"   ✓ Found PageID from metadata: {page_id}")
            else:
                print("   ❌ No PageID found.")
                return
        else:
            print("   ❌ No metadata returned.")
            return
    
    # 3. Now fetch the text using the corrected function
    print(f"\n4. Fetching text from page {page_id} using GetPageMetadata...")
    text = get_page_text(page_id)
    
    if text:
        print(f"\n✅ Success! Found {len(text)} characters of text.")
        print("\nFirst 500 characters:")
        print("-" * 40)
        print(text[:500])
        print("-" * 40)
        print("\n✅ API fix is working!")
    else:
        print("\n❌ No text returned. The API may still have issues.")


if __name__ == "__main__":
    test_api_fix()