"""
BHL NER Dashboard
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
    st.session_state.ner_processor = NERProcessor()

# Initialise storage for processed text and entities
if "processed_text" not in st.session_state:
    st.session_state.processed_text = ""
if "entities" not in st.session_state:
    st.session_state.entities = []

# After NER processor loads, print the time taken
if "ner_processor" not in st.session_state:
    print("Creating NER processor...")
    load_start = time.time()
    st.session_state.ner_processor = NERProcessor()
    load_end = time.time()
    print(f"✓ NER processor loaded in {load_end - load_start:.2f} seconds")
else:
    print("✓ NER processor already loaded from session state")

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

if "fetch_clicked" not in st.session_state:
    st.session_state.fetch_clicked = False
if "page_id_value" not in st.session_state:
    st.session_state.page_id_value = ""

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

# Initialise variables to store input
page_id = ""
pasted_text = ""

# Create two columns for the two input methods
col1, col2 = st.columns(2)

# --- Column 1: BHL Page ID ---
with col1:
    st.write("**Option 1: Fetch from BHL**")
    page_id = st.text_input(
        "Enter a BHL Page ID:",
        placeholder="e.g., 53548146",
        help="Find a Page ID by opening a page on biodiversitylibrary.org and copying the number from the URL",
        key="page_id_input"
    )
    
    fetch_button = st.button("Fetch from BHL", use_container_width=True, key="fetch_button")
    
    # NEW: Store whether the button was clicked
    if fetch_button:
        st.session_state.fetch_clicked = True
        st.session_state.page_id_value = page_id
    elif not page_id.strip():
        st.session_state.fetch_clicked = False
    
    # NEW: Show message if button was clicked OR if Page ID is entered and we're showing the display
    if fetch_button:
        if page_id.strip():
            st.info(f"⏳ Fetching BHL page {page_id}... (Coming in Week 5)")
        else:
            st.warning("⚠️ Please enter a Page ID.")
    elif page_id.strip():
        # Optional: Show a subtle hint that Enter was pressed
        pass

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
            st.info(f"⏳ Processing {len(pasted_text)} characters... (Coming in Week 5)")
        else:
            st.warning("⚠️ Please paste some text.")

st.divider()

# ============================================================================
# TEXT DISPLAY SECTION
# ============================================================================

st.subheader("📄 Original Text")

# Determine which text to display
# Priority: If user pasted text, show that. If user entered a Page ID, show that.
display_text = None
source = None

# Check what the user has entered
has_pasted_text = pasted_text and pasted_text.strip()
has_page_id = page_id and page_id.strip()

if has_pasted_text:
    display_text = pasted_text
    source = "Pasted text"
elif has_page_id:
    display_text = f"Text from BHL Page ID: {page_id}\n\n(Your fetched text will appear here when connected in Week 5)"
    source = f"BHL Page {page_id}"
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
if has_pasted_text:
    char_count = len(display_text)
    st.caption(f"📊 Character count: {char_count}")
elif has_page_id:
    char_count = len(display_text)
    st.caption(f"📊 Character count: {char_count}")

st.divider()


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("🌿 BHL NER Dashboard | Built with Streamlit")