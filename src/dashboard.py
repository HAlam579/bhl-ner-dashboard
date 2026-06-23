"""
BHL NER Dashboard
Week 5: Connected Backend to Dashboard
"""

import sys
import os
import streamlit as st
import time

start_time = time.time()

print("=" * 60)
print("BHL NER Dashboard - Starting up...")
print("=" * 60)

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
                    st.session_state.processed_text = text
                    st.session_state.entities = []
                    st.session_state.counts = {}
                    st.session_state.source = f"BHL Page {page_id}"
                    st.success(f"✅ Successfully fetched {len(text)} characters from page {page_id}!")
                else:
                    st.error(f"❌ Failed to fetch text from page {page_id}. Please check the Page ID.")
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

st.divider()

# ============================================================================
# TEXT DISPLAY SECTION
# ============================================================================

st.subheader("📄 Original Text")

# Get the stored text and source from session state
stored_text = st.session_state.get("processed_text", "")
stored_source = st.session_state.get("source", "Waiting for input...")

if stored_text:
    display_text = stored_text
    source = stored_source
else:
    display_text = "Enter a Page ID or paste some text above. The text will appear here."
    source = "Waiting for input..."

# Display the text in a text box
st.text_area(
    label=f"**Source:** {source}",
    value=display_text,
    height=200,
    disabled=True,
    key="display_text_area"
)

# Show character count if there is actual text
if stored_text:
    char_count = len(stored_text)
    st.caption(f"📊 Character count: {char_count}")

st.divider()

# ============================================================================
# ENTITIES DISPLAY (Task 3 will add highlighting)
# ============================================================================

st.subheader("🔍 Extracted Entities")

entities = st.session_state.get("entities", [])
counts = st.session_state.get("counts", {})

if entities:
    st.write("**Entity type counts:**")
    for label, count in counts.items():
        st.write(f"  - {label}: {count}")
    
    st.write("**List of entities found:**")
    for entity in entities:
        source_info = f" ({entity.get('source', 'N/A')})" if entity.get('source') else ""
        score_info = f" - score: {entity.get('score', 'N/A'):.2f}" if entity.get('score') else ""
        st.write(f"  - {entity['text']} -> {entity['label']}{source_info}{score_info}")
else:
    st.info("💡 No entities found yet. Paste text or fetch a BHL page and click a button above.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("🌿 BHL NER Dashboard | Built with Streamlit")