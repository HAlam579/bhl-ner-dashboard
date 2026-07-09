import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import time

start_time = time.time()

print("=" * 60)
print("BHL NER Dashboard - Starting up...")
print("=" * 60)


ENTITY_COLOURS = {
    "SPECIES": "#90EE90",
    "LOCATION": "#00FFFF",
    "DATE": "#FFD700",
    "PERSON": "#FFB6C1",
    "ORGANIZATION": "#DDA0DD",
    "HABITAT": "#D2A679",
    "ORGANISM": "#AAFF00",
}
DEFAULT_COLOUR = "#FFFF00"


def highlight_entities(text, entities):
    """Highlight entities in text with background colours."""
    if not entities or len(entities) == 0:
        return text
    
    sorted_entities = sorted(entities, key=lambda x: len(x["text"]), reverse=True)
    replacements = []
    
    for entity in sorted_entities:
        entity_text = entity["text"]
        label = entity["label"]
        
        if len(entity_text) < 2:
            continue
        
        colour = ENTITY_COLOURS.get(label, DEFAULT_COLOUR)
        start_idx = text.lower().find(entity_text.lower())
        
        if start_idx != -1:
            end_idx = start_idx + len(entity_text)
            
            overlap = False
            for existing_start, existing_end, _ in replacements:
                if not (end_idx <= existing_start or start_idx >= existing_end):
                    overlap = True
                    break
            
            if not overlap:
                html_span = f'<span style="background-color: {colour}; padding: 2px 4px; border-radius: 3px;" title="{label}">{entity_text}</span>'
                replacements.append((start_idx, end_idx, html_span))
    
    replacements.sort(key=lambda x: x[0], reverse=True)
    
    highlighted_text = text
    for start_idx, end_idx, html_span in replacements:
        highlighted_text = highlighted_text[:start_idx] + html_span + highlighted_text[end_idx:]
    
    return highlighted_text


project_root = os.path.dirname(os.path.abspath(__file__))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.fetch_bhl import get_page_text, search_publications
from src.ner_processor import NERProcessor

if "ner_processor" not in st.session_state:
    print("Creating NER processor...")
    load_start = time.time()
    st.session_state.ner_processor = NERProcessor()
    load_end = time.time()
    print(f"✓ NER processor loaded in {load_end - load_start:.2f} seconds")
else:
    print("✓ NER processor already loaded from session state")

if "processed_text" not in st.session_state:
    st.session_state.processed_text = ""
if "entities" not in st.session_state:
    st.session_state.entities = []
if "counts" not in st.session_state:
    st.session_state.counts = {}
if "source" not in st.session_state:
    st.session_state.source = ""

end_time = time.time()
print(f"✓ Total startup time: {end_time - start_time:.2f} seconds")
print("=" * 60)

st.set_page_config(
    page_title="BHL NER Dashboard",
    page_icon="🌿",
    layout="wide"
)


# SIDEBAR

with st.sidebar:
    st.subheader("⚙️ Extraction Method")
    extraction_method = st.selectbox(
        "Choose the NER extraction method:",
        ["Hybrid (spaCy + GLiNER2)", "spaCy Only", "GLiNER2 Only"],
        help="Hybrid uses both spaCy and GLiNER2 for better accuracy. Choose spaCy Only or GLiNER2 Only to test individual models.",
        key="extraction_method_selectbox"
    )

    st.divider()

    st.subheader("🎨 Legend")
    for label, colour in ENTITY_COLOURS.items():
        st.markdown(
            f'<span style="background-color: {colour}; padding: 2px 8px; border-radius: 3px;">{label}</span>',
            unsafe_allow_html=True
        )


# HEADER

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


# INPUT SECTION

st.subheader("📝 Input")

col1, col2 = st.columns(2)

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
                    st.session_state.source = f"BHL Page {page_id}"
                    
                    processor = st.session_state.ner_processor
                    results = processor.process_text(text, method=extraction_method)
                    st.session_state.entities = results["entities"]
                    st.session_state.counts = results["counts"]
                    
                    st.success(f"✅ Fetched {len(text)} characters and found {results['total']} entities!")
                else:
                    st.error(f"❌ Failed to fetch text from page {page_id}.")
        else:
            st.warning("⚠️ Please enter a Page ID.")

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
                results = processor.process_text(pasted_text, method=extraction_method)
                st.session_state.processed_text = pasted_text
                st.session_state.entities = results["entities"]
                st.session_state.counts = results["counts"]
                st.session_state.source = "Pasted text"
                
                st.success(f"✅ Found {results['total']} entities in {len(pasted_text)} characters!")
        else:
            st.warning("⚠️ Please paste some text.")

col_clear1, col_clear2, col_clear3 = st.columns([1, 1, 4])
with col_clear1:
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.processed_text = ""
        st.session_state.entities = []
        st.session_state.counts = {}
        st.session_state.source = ""

st.divider()


# RESULTS SECTION

stored_text = st.session_state.get("processed_text", "")
entities = st.session_state.get("entities", [])
counts = st.session_state.get("counts", {})

if stored_text and entities:
    

    st.subheader("🎨 Highlighted Text")
    st.caption(f"Source: {st.session_state.get('source', 'Unknown')}")
    st.caption(f"Character count: {len(stored_text)}")
    
    highlighted_html = highlight_entities(stored_text, entities)
    st.markdown(highlighted_html, unsafe_allow_html=True)
    
    st.divider()
    

    st.subheader("🔍 Extracted Entities")
    
    # Entity Table
    df_data = []
    for entity in entities:
        row = {
            "Entity": entity["text"],
            "Type": entity["label"],
            "Source": entity.get("source", "N/A"),
            "Start": entity.get("start", "N/A"),
            "End": entity.get("end", "N/A"),
        }
        # Only add Score if it exists
        if "score" in entity:
            row["Score"] = entity.get("score", "N/A")
        df_data.append(row)

    df = pd.DataFrame(df_data).drop_duplicates(subset=["Entity", "Type"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    

    st.write("**Summary:**")
    
    tags_html = ""
    for label, count in counts.items():
        colour = ENTITY_COLOURS.get(label, DEFAULT_COLOUR)
        tags_html += f'<span style="background-color: {colour}; padding: 4px 12px; border-radius: 12px; font-weight: bold; display: inline-block; margin: 4px 4px;">{label}: {count}</span> '
    
    st.markdown(tags_html, unsafe_allow_html=True)
    
    # Bar Chart
    st.subheader("📊 Entity Type Distribution")

    if counts:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(counts.keys(), counts.values(), color='#00897B')
        ax.set_xlabel('Entity Type')
        ax.set_ylabel('Count')
        ax.set_title('Entities Found')
        ax.tick_params(axis='x', rotation=45)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        st.pyplot(fig)
    else:
        st.info("No data to display.")

    # CSV Download
    st.write("**Download Results:**")
    
    df_export = pd.DataFrame(df_data)
    csv_buffer = io.StringIO()
    df_export.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="⬇️ Download as CSV",
        data=csv_data,
        file_name="bhl_ner_entities.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.info("💡 No text processed yet. Paste text or fetch a BHL page using the buttons above.")



# FOOTER

st.divider()
st.caption("🌿 BHL NER Dashboard | Built with Streamlit")