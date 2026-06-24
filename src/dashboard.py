"""
BHL NER Dashboard
Week 5: Connected Backend to Dashboard
"""

import sys
import os
import streamlit as st
import pandas as pd
import time

start_time = time.time()

print("=" * 60)
print("BHL NER Dashboard - Starting up...")
print("=" * 60)

# ============================================================================
# HIGHLIGHTING FUNCTION
# ============================================================================

def highlight_entities(text, entities):
    """
    Highlight entities in text with background colours.
    
    Args:
        text (str): The original text
        entities (list): List of entities with 'text' and 'label'
    
    Returns:
        str: HTML string with highlighted entities
    """
    # If no entities, return the text as plain
    if not entities or len(entities) == 0:
        return text
    
    # Define colours for different entity types
    colours = {
        "SPECIES": "#90EE90",      # Light green
        "GPE": "#87CEEB",          # Light blue
        "LOCATION": "#87CEEB",     # Light blue
        "DATE": "#FFD700",         # Gold
        "PERSON": "#FFB6C1",       # Light pink
        "ORG": "#DDA0DD",          # Plum
        "HABITAT": "#98FB98",      # Pale green
        "ORGANISM": "#AFEEEE",     # Pale turquoise
        "TAXON": "#90EE90",        # Light green
    }
    default_colour = "#FFFF00"     # Yellow
    
    # Sort entities by length (longest first) to handle overlapping
    sorted_entities = sorted(entities, key=lambda x: len(x["text"]), reverse=True)
    
    # Create a list of replacements to apply
    replacements = []
    
    for entity in sorted_entities:
        entity_text = entity["text"]
        label = entity["label"]
        
        # Skip very short entities (less than 2 characters)
        if len(entity_text) < 2:
            continue
        
        colour = colours.get(label, default_colour)
        
        # Find the entity in the text (case insensitive)
        start_idx = text.lower().find(entity_text.lower())
        if start_idx != -1:
            end_idx = start_idx + len(entity_text)
            
            # Check if this position overlaps with an existing replacement
            overlap = False
            for existing_start, existing_end, _ in replacements:
                if not (end_idx <= existing_start or start_idx >= existing_end):
                    overlap = True
                    break
            
            if not overlap:
                # Create the HTML span
                html_span = f'<span style="background-color: {colour}; padding: 2px 4px; border-radius: 3px;" title="{label}">{entity_text}</span>'
                replacements.append((start_idx, end_idx, html_span))
    
    # Sort replacements by start index (so we replace from the end)
    replacements.sort(key=lambda x: x[0], reverse=True)
    
    # Apply replacements from end to start
    highlighted_text = text
    for start_idx, end_idx, html_span in replacements:
        highlighted_text = highlighted_text[:start_idx] + html_span + highlighted_text[end_idx:]
    
    return highlighted_text

project_root = os.path.dirname(os.path.abspath(__file__))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.fetch_bhl import get_page_text, search_publications
from src.ner_processor import NERProcessor

# Initialise the NER processor once (models load once, not every button click)
if "ner_processor" not in st.session_state:
    print("Creating NER processor...")
    load_start = time.time()
    st.session_state.ner_processor = NERProcessor()
    load_end = time.time()
    print(f"✓ NER processor loaded in {load_end - load_start:.2f} seconds")
else:
    print("✓ NER processor already loaded from session state")

# Initialise storage for processed text and entities
if "processed_text" not in st.session_state:
    st.session_state.processed_text = ""
if "entities" not in st.session_state:
    st.session_state.entities = []
if "counts" not in st.session_state:
    st.session_state.counts = {}
if "source" not in st.session_state:
    st.session_state.source = ""

# Print total startup time
end_time = time.time()
print(f"✓ Total startup time: {end_time - start_time:.2f} seconds")
print("=" * 60)

# Page configuration
st.set_page_config(
    page_title="BHL NER Dashboard",
    page_icon="🌿",
    layout="wide"
)



# ============================================================================
# SIDEBAR: LEGEND (Always accessible from anywhere)
# ============================================================================

with st.sidebar:
    st.subheader("🎨 Legend")
    colours = {
        "SPECIES": "#90EE90",
        "GPE": "#89CFF0",
        "LOCATION": "#00FFFF",
        "DATE": "#FFD700",
        "PERSON": "#FFB6C1",
        "ORG": "#DDA0DD",
        "HABITAT": "#D2A679",
        "ORGANISM": "#AAFF00",
    }
    for label, colour in colours.items():
        st.markdown(f'<span style="background-color: {colour}; padding: 2px 8px; border-radius: 3px;">{label}</span>', unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.title("🌿 BHL NER Dashboard")
st.markdown("""
This tool uses **Named Entity Recognition (NER)** to extract:
- 🧬 **Species names**
- 📍 **Locations** 
- 📅 **Dates** 
- 👤 **People** 

Enter a BHL Page ID or paste text below to get started.
""")

st.divider()

# ============================================================================
# INPUT SECTION
# ============================================================================

st.subheader("📝 Input")

# Create two columns for the two input methods
col1, col2 = st.columns(2)

# --- Column 1: BHL Page ID ---
with col1:
    st.write("**Option 1: Fetch from BHL**")
    page_id = st.text_input(
        "Enter a BHL Page ID:",
        placeholder="e.g., 54850898",
        help="Find a Page ID by opening a page on biodiversitylibrary.org and copying the number from the URL",
        key="page_id_input"
    )
    
    fetch_button = st.button("Fetch from BHL", use_container_width=True, key="fetch_button")
    
    if fetch_button:
        if page_id.strip():
            with st.spinner(f"📡 Fetching BHL page {page_id}..."):
                text = get_page_text(page_id)
                if text:
                    # Fetch the text
                    st.session_state.processed_text = text
                    st.session_state.source = f"BHL Page {page_id}"
                    
                    # Auto-process with NER
                    processor = st.session_state.ner_processor
                    results = processor.process_text(text)
                    st.session_state.entities = results["entities"]
                    st.session_state.counts = results["counts"]
                    
                    st.success(f"✅ Fetched {len(text)} characters and found {results['total']} entities!")
                else:
                    st.error(f"❌ Failed to fetch text from page {page_id}.")
        else:
            st.warning("⚠️ Please enter a Page ID.")

# --- Column 2: Paste Text ---
with col2:
    st.write("**Option 2: Paste your own text**")
    pasted_text = st.text_area(
        "Paste biodiversity text here:",
        height=150,
        placeholder="Paste any text containing species names, locations, or dates...",
        key="pasted_text_input"
    )
    
    process_button = st.button("Process Text", use_container_width=True, key="process_button")
    
    if process_button:
        if pasted_text.strip():
            with st.spinner(f"🧠 Processing {len(pasted_text)} characters..."):
                processor = st.session_state.ner_processor
                results = processor.process_text(pasted_text)
                
                st.session_state.processed_text = pasted_text
                st.session_state.entities = results["entities"]
                st.session_state.counts = results["counts"]
                st.session_state.source = "Pasted text"
                
                st.success(f"✅ Found {results['total']} entities in {len(pasted_text)} characters!")
        else:
            st.warning("⚠️ Please paste some text.")

# Clear BTN
col_clear1, col_clear2, col_clear3 = st.columns([1, 1, 4])
with col_clear1:
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.processed_text = ""
        st.session_state.entities = []
        st.session_state.counts = {}
        st.session_state.source = ""
        st.rerun()

st.divider()

# ============================================================================
# RESULTS SECTION
# ============================================================================

# Get the stored text and entities from session state
stored_text = st.session_state.get("processed_text", "")
entities = st.session_state.get("entities", [])
counts = st.session_state.get("counts", {})

if stored_text and entities:
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🎨 Highlighted Text")
        st.caption(f"Source: {st.session_state.get('source', 'Unknown')}")
        st.caption(f"Character count: {len(stored_text)}")


        # Legend
        with st.popover("🎨 Legend"):
            st.write("**Entity Type Colours:**")
            colours = {
                "SPECIES": "#90EE90",
                "GPE": "#87CEEB",
                "LOCATION": "#87CEEB",
                "DATE": "#FFD700",
                "PERSON": "#FFB6C1",
                "ORG": "#DDA0DD",
                "HABITAT": "#98FB98",
                "ORGANISM": "#AFEEEE",
            }
            for label, colour in colours.items():
                if any(e["label"] == label for e in entities):
                    st.markdown(f'<span style="background-color: {colour}; padding: 2px 8px; border-radius: 3px;">{label}</span>', unsafe_allow_html=True)
        
        # Highlight the entities in the text
        highlighted_html = highlight_entities(stored_text, entities)
        
        # Display the highlighted text
        st.markdown(highlighted_html, unsafe_allow_html=True)

    with col_right:
        # Simplified entity list as a table
        st.subheader("🔍 Extracted Entities")
        
        # Create a clean DataFrame
        df_data = []
        for entity in entities:
            df_data.append({
                "Entity": entity["text"],
                "Type": entity["label"],
                "Source": entity.get("source", "N/A"),
            })
        
        # Remove duplicates
        df = pd.DataFrame(df_data).drop_duplicates(subset=["Entity", "Type"])
        
        # Display as a table
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Show summary counts
        st.write("**Summary:**")
        for label, count in counts.items():
            st.write(f"  - {label}: {count}")

else:
    st.info("💡 No text processed yet. Paste text or fetch a BHL page using the buttons above.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("🌿 BHL NER Dashboard | Built with Streamlit")