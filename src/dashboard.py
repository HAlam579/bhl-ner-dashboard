import sys
import os
import warnings
import logging

# === SUPPRESS ALL TERMINAL NOISE ===

warnings.filterwarnings("ignore")

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.getLogger("streamlit.watcher").setLevel(logging.ERROR)
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

logging.getLogger("transformers").setLevel(logging.ERROR)

logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)

# Redirect stderr to suppress transformer module scanning
import contextlib
import io

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime

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


def show_model_loading_status(processor, extraction_method):
    """
    Show which models were loaded during the request.
    Returns: A tuple of (loaded_models_list, all_loaded_boolean)
    """
    loaded_models = []
    if processor.models_loaded["spacy"]:
        loaded_models.append("spaCy")
    if processor.models_loaded["gliner2"]:
        loaded_models.append("GLiNER2")
    if processor.models_loaded["gliner"]:
        loaded_models.append("GLiNER")
    if processor.models_loaded["gnfinder"]:
        loaded_models.append("GNfinder")
    
    # Check if all expected models for the selected method are loaded
    all_loaded = True
    if "spaCy" in extraction_method and not processor.models_loaded["spacy"]:
        all_loaded = False
    if "GLiNER2" in extraction_method and not processor.models_loaded["gliner2"]:
        all_loaded = False
    if "GLiNER" in extraction_method and not processor.models_loaded["gliner"]:
        all_loaded = False
    if "GNfinder" in extraction_method and not processor.models_loaded["gnfinder"]:
        all_loaded = False
    
    return loaded_models, all_loaded


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
    print(f"✓ NER processor created in {load_end - load_start:.2f} seconds")
    st.session_state.processor_load_time = load_end - load_start
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
if "processing_times" not in st.session_state:
    st.session_state.processing_times = []
if "total_pages_processed" not in st.session_state:
    st.session_state.total_pages_processed = 0

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
        [
            "Hybrid (spaCy + GLiNER2)",
            "Hybrid (spaCy + GLiNER)",
            "Hybrid (spaCy + GLiNER + GNfinder)",
            "spaCy Only",
            "GLiNER2 Only",
            "GLiNER Only",
            "GNfinder Only"
        ],
        help="Hybrid combines models for better accuracy. GLiNER is an updated model with better zero-shot capabilities.",
        key="extraction_method_selectbox"
    )

    st.divider()

    st.subheader("🎨 Legend")
    for label, colour in ENTITY_COLOURS.items():
        st.markdown(
            f'<span style="background-color: {colour}; padding: 2px 8px; border-radius: 3px;">{label}</span>',
            unsafe_allow_html=True
        )

    st.divider()

    # Performance Metrics in Sidebar
    st.subheader("⚡ Performance Metrics")
    
    # Processor load time
    processor_load = st.session_state.get("processor_load_time", 0)
    st.metric(
        "Model Loading Time",
        f"{processor_load:.2f}s",
        help="Time taken to load NER models (only on first use)"
    )
    
    # Last processing time
    if st.session_state.processing_times:
        last_time = st.session_state.processing_times[-1]
        st.metric(
            "Last Processing Time",
            f"{last_time:.2f}s",
            help="Time taken for the most recent text processing"
        )
        
        # Average processing time
        avg_time = sum(st.session_state.processing_times) / len(st.session_state.processing_times)
        st.metric(
            "Average Processing Time",
            f"{avg_time:.2f}s",
            help="Average time across all processed texts"
        )
        
        # Total processed
        st.metric(
            "Total Processed",
            st.session_state.total_pages_processed,
            help="Total number of texts processed"
        )
    
    st.divider()
    
    # Model status
    st.subheader("📊 Model Status")
    processor = st.session_state.ner_processor

    if processor.models_loaded["spacy"]:
        st.success("✅ spaCy: Loaded")
    else:
        st.info("⏳ spaCy: Not loaded")

    if processor.models_loaded["gliner2"]:
        st.success("✅ GLiNER2: Loaded")
    else:
        st.info("⏳ GLiNER2: Not loaded")

    if processor.models_loaded["gliner"]:
        st.success("✅ GLiNER: Loaded")
    else:
        st.info("⏳ GLiNER: Not loaded")

    if processor.models_loaded["gnfinder"]:
        st.success("✅ GNfinder: Loaded")
    else:
        st.info("⏳ GNfinder: Not loaded")


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


# ADD USER INSTRUCTIONS
with st.expander("📖 How to Use This Dashboard", expanded=False):
    st.markdown("""
    ### Quick Start
    1. **Fetch from BHL**: Enter a Page ID (e.g., `54850898`) and click "Fetch from BHL"
    2. **Or paste text**: Paste your own biodiversity text in the text area
    3. **Choose a method**: Select an extraction method from the sidebar
    4. **View results**: See highlighted text, entity table, and charts
    5. **Export**: Download results as CSV
    
    ### 💡 Tips
    - **Best overall**: Use "Hybrid (spaCy + GLiNER + GNfinder)" for maximum coverage
    - **Real confidence scores**: GLiNER and GNfinder provide real confidence scores
    - **First request may be slow**: Models load on first use, subsequent requests are faster
    - **Large texts**: Processing long texts may take a few seconds
    
    ### 📊 Entity Types
    - 🟢 **SPECIES**: Scientific and common species names
    - 🔵 **LOCATION**: Places, regions, countries
    - 🟡 **DATE**: Years, dates, time periods
    - 🩷 **PERSON**: People, authors, collectors
    - 🟣 **ORGANIZATION**: Institutions, societies, projects
    - 🟠 **HABITAT**: Environments, ecosystems, habitats
    - 🟢 **ORGANISM**: Plants, animals, fungi (non-species)
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
                # Track fetch time
                fetch_start = time.time()
                text = get_page_text(page_id)
                fetch_time = time.time() - fetch_start
                
                if text:
                    st.session_state.processed_text = text
                    st.session_state.source = f"BHL Page {page_id}"
                    
                    # Track processing time
                    process_start = time.time()
                    processor = st.session_state.ner_processor
                    results = processor.process_text(text, method=extraction_method)
                    process_time = time.time() - process_start
                    
                    st.session_state.entities = results["entities"]
                    st.session_state.counts = results["counts"]
                    st.session_state.processing_times.append(process_time)
                    st.session_state.total_pages_processed += 1
                    
                    # Show timing information in success message
                    st.success(
                        f"✅ Fetched {len(text)} characters and found {results['total']} entities!\n\n"
                        f"⏱️ **Fetch time**: {fetch_time:.2f}s | "
                        f"**Processing time**: {process_time:.2f}s | "
                        f"**Total**: {fetch_time + process_time:.2f}s"
                    )
                    
                    # Show model loading status (using the reusable function)
                    loaded_models, all_loaded = show_model_loading_status(processor, extraction_method)
                    
                    if not all_loaded and loaded_models:
                        st.info(f"💡 Models ({', '.join(loaded_models)}) were loaded during this request. Subsequent requests will be faster.")
                    elif not all_loaded:
                        st.info("💡 Models are loading... Subsequent requests will be faster.")
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
                # Track processing time
                process_start = time.time()
                processor = st.session_state.ner_processor
                results = processor.process_text(pasted_text, method=extraction_method)
                process_time = time.time() - process_start
                
                st.session_state.processed_text = pasted_text
                st.session_state.entities = results["entities"]
                st.session_state.counts = results["counts"]
                st.session_state.source = "Pasted text"
                st.session_state.processing_times.append(process_time)
                st.session_state.total_pages_processed += 1
                
                st.success(
                    f"✅ Found {results['total']} entities in {len(pasted_text)} characters!\n\n"
                    f"⏱️ **Processing time**: {process_time:.2f}s"
                )
                
                # Show model loading status (using the reusable function)
                loaded_models, all_loaded = show_model_loading_status(processor, extraction_method)
                
                if not all_loaded and loaded_models:
                    st.info(f"💡 Models ({', '.join(loaded_models)}) were loaded during this request. Subsequent requests will be faster.")
                elif not all_loaded:
                    st.info("💡 Models are loading... Subsequent requests will be faster.")
        else:
            st.warning("⚠️ Please paste some text.")

col_clear1, col_clear2, col_clear3 = st.columns([1, 1, 4])
with col_clear1:
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.processed_text = ""
        st.session_state.entities = []
        st.session_state.counts = {}
        st.session_state.source = ""
        st.session_state.processing_times = []
        st.session_state.total_pages_processed = 0

st.divider()


# RESULTS SECTION

stored_text = st.session_state.get("processed_text", "")
entities = st.session_state.get("entities", [])
counts = st.session_state.get("counts", {})

if stored_text and entities:
    
    # Show timing summary at top of results
    if st.session_state.processing_times:
        col_time1, col_time2, col_time3 = st.columns(3)
        with col_time1:
            st.metric(
                "🕐 Processing Time",
                f"{st.session_state.processing_times[-1]:.2f}s",
                help="Time taken for this text processing"
            )
        with col_time2:
            avg_time = sum(st.session_state.processing_times) / len(st.session_state.processing_times)
            st.metric(
                "📊 Average Time",
                f"{avg_time:.2f}s",
                help="Average processing time across all texts"
            )
        with col_time3:
            st.metric(
                "📄 Total Processed",
                st.session_state.total_pages_processed,
                help="Total number of texts processed in this session"
            )
    
    st.divider()

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