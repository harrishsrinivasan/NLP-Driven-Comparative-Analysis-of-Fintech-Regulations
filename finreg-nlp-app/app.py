"""
FinReg NLP Analysis - Streamlit Application
Main application for comparative analysis of financial technology regulations
"""

import os
import streamlit as st

# Set environment variable before importing other modules
os.environ["USE_TF"] = "0"

# Import modules
from modules.pdf_processor import (
    load_spacy_model,
    process_uploaded_file,
    tokenize_text,
    get_text_statistics
)
from modules.frequency_analyzer import (
    gen_freq,
    display_frequency_analysis,
    compare_word_frequencies
)
from modules.similarity_analyzer import (
    load_bert_model,
    compute_tfidf_similarity,
    compute_bert_similarity,
    display_similarity_analysis,
    compare_similarity_methods
)
from modules.sentiment_analyzer import (
    analyze_document_sentiment,
    display_sentiment_analysis,
    display_sentiment_comparison,
    get_sentiment_summary
)
from modules.data_manager import (
    initialize_session_state,
    validate_uploaded_files,
    store_document_data,
    store_similarity_results,
    store_sentiment_result,
    has_processed_documents,
    get_document_names,
    get_all_sentiment_results_list,
    clear_all_cache,
    is_processing_complete,
    set_processing_complete
)
from utils.config import (
    APP_TITLE,
    APP_SUBTITLE,
    SIDEBAR_HEADER,
    MAX_FILE_UPLOADS,
    MSG_UPLOADING,
    MSG_EXTRACTING,
    MSG_CLEANING,
    MSG_ANALYZING_FREQ,
    MSG_COMPUTING_TFIDF,
    MSG_COMPUTING_BERT,
    MSG_ANALYZING_SENTIMENT,
    MSG_COMPLETE
)


# Page configuration
st.set_page_config(
    page_title="FinReg NLP Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_models():
    """Load all required models at startup."""
    with st.spinner("🔄 Loading models (this may take a moment)..."):
        spacy_model = load_spacy_model()
        bert_model = load_bert_model()
    return spacy_model, bert_model


def process_documents(uploaded_files, nlp_model):
    """
    Process all uploaded documents through the NLP pipeline.
    
    Args:
        uploaded_files: List of uploaded file objects
        nlp_model: Loaded spaCy model
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(uploaded_files)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (idx + 1) / total_files
        progress_bar.progress(progress)
        status_text.text(f"Processing {uploaded_file.name} ({idx + 1}/{total_files})...")
        
        # Extract and clean text
        with st.spinner(f"{MSG_EXTRACTING} {uploaded_file.name}"):
            result = process_uploaded_file(uploaded_file)
        
        if not result['success']:
            st.warning(f"⚠️ Skipping {uploaded_file.name} - could not extract text")
            continue
        
        doc_name = uploaded_file.name.replace('.pdf', '')
        raw_text = result['raw_text']
        cleaned_text = result['cleaned_text']
        
        # Tokenize and generate frequencies
        with st.spinner(f"{MSG_ANALYZING_FREQ}"):
            doc = tokenize_text(cleaned_text, nlp_model)
            if doc:
                word_freq = gen_freq(doc)
                store_document_data(doc_name, raw_text, cleaned_text, word_freq)
    
    progress_bar.progress(1.0)
    status_text.text(MSG_COMPLETE)
    
    # Clear progress indicators after a moment
    import time
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()


def compute_all_similarities():
    """Compute TF-IDF and BERT similarities."""
    docs_dict = st.session_state.cleaned_texts
    
    if len(docs_dict) < 2:
        st.warning("⚠️ Need at least 2 documents for similarity analysis")
        return
    
    # TF-IDF Similarity
    with st.spinner(MSG_COMPUTING_TFIDF):
        tfidf_sim, _, _ = compute_tfidf_similarity(docs_dict)
    
    # BERT Similarity
    bert_model = load_bert_model()
    with st.spinner(MSG_COMPUTING_BERT):
        bert_sim, _ = compute_bert_similarity(bert_model, docs_dict)
    
    # Store results
    store_similarity_results(tfidf_sim, bert_sim)


def compute_all_sentiments():
    """Compute sentiment analysis for all documents."""
    raw_texts = st.session_state.raw_texts
    
    with st.spinner(MSG_ANALYZING_SENTIMENT):
        for doc_name, raw_text in raw_texts.items():
            sentiment_result = analyze_document_sentiment(raw_text, doc_name)
            if sentiment_result:
                store_sentiment_result(doc_name, sentiment_result)


def main():
    """Main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title(APP_TITLE)
    st.markdown(f"*{APP_SUBTITLE}*")
    st.markdown("---")
    
    # Sidebar - File Upload
    with st.sidebar:
        st.header(SIDEBAR_HEADER)
        st.markdown(f"Upload up to **{MAX_FILE_UPLOADS} PDF documents** for analysis")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help=f"Select up to {MAX_FILE_UPLOADS} PDF files containing regulatory documents"
        )
        
        st.markdown("---")
        
        # Process button
        if uploaded_files:
            st.info(f"📄 {len(uploaded_files)} file(s) selected")
            
            # Validate files
            is_valid, error_msg = validate_uploaded_files(uploaded_files)
            
            if not is_valid:
                st.error(error_msg)
            else:
                if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                    # Clear previous results
                    clear_all_cache()
                    
                    # Load models
                    nlp_model, bert_model = load_models()
                    
                    # Process documents
                    st.markdown("### Processing Pipeline")
                    process_documents(uploaded_files, nlp_model)
                    
                    # Compute similarities
                    if len(st.session_state.cleaned_texts) >= 2:
                        compute_all_similarities()
                    
                    # Compute sentiments
                    compute_all_sentiments()
                    
                    # Mark processing as complete
                    set_processing_complete(True)
                    
                    st.success("✅ All analyses complete!")
                    st.rerun()
        
        else:
            st.info("👆 Please upload PDF files to begin")
        
        st.markdown("---")
        
        # Clear cache button
        if has_processed_documents():
            if st.button("🗑️ Clear All Data", use_container_width=True):
                clear_all_cache()
                st.rerun()
        
        # Info section
        st.markdown("---")
        st.markdown("### 📋 Analysis Includes:")
        st.markdown("""
        - **Word Frequency**: Most common terms
        - **WordClouds**: Visual word representation
        - **TF-IDF**: Term-based similarity
        - **BERT**: Semantic similarity
        - **Sentiment**: Emotional tone analysis
        """)
    
    # Main content area
    if not is_processing_complete():
        # Welcome message
        st.info("👈 Upload PDF documents using the sidebar to get started")
        
        st.markdown("## 🎯 How to Use")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Step 1: Upload Documents
            - Click the file uploader in the sidebar
            - Select up to 4 PDF files
            - Files should contain regulatory text
            
            ### Step 2: Process
            - Click "Process Documents"
            - Wait for the NLP pipeline to complete
            - This may take 30-60 seconds
            """)
        
        with col2:
            st.markdown("""
            ### Step 3: Explore Results
            - Navigate through the tabs below
            - View frequency analysis and word clouds
            - Compare document similarities
            - Analyze sentiment distributions
            
            ### Step 4: Export (Optional)
            - Download results using download buttons
            - Export charts and data tables
            """)
        
        st.markdown("---")
        st.markdown("### 📊 Sample Analyses")
        st.markdown("Once you process documents, you'll see detailed analysis in these tabs:")
        
        # Show sample tabs (disabled)
        tab1, tab2, tab3 = st.tabs(["📈 Frequency Analysis", "🔍 Similarity Analysis", "😊 Sentiment Analysis"])
        
        with tab1:
            st.info("Upload and process documents to view word frequency analysis and word clouds")
        
        with tab2:
            st.info("Upload and process documents to view TF-IDF and BERT similarity matrices")
        
        with tab3:
            st.info("Upload and process documents to view sentiment distribution analysis")
    
    else:
        # Display results in tabs
        st.success(f"✅ Successfully analyzed {len(get_document_names())} document(s)")
        
        tab1, tab2, tab3 = st.tabs(["📈 Frequency Analysis", "🔍 Similarity Analysis", "😊 Sentiment Analysis"])
        
        # Tab 1: Frequency Analysis
        with tab1:
            st.header("Word Frequency Analysis")
            st.markdown("Explore the most frequently occurring words in each document")
            st.markdown("---")
            
            doc_names = get_document_names()
            
            for doc_name in doc_names:
                word_freq = st.session_state.word_frequencies[doc_name]
                display_frequency_analysis(doc_name, word_freq)
                st.markdown("---")
            
            # Comparison section
            if len(doc_names) > 1:
                st.subheader("📊 Cross-Document Comparison")
                comparison_df = compare_word_frequencies(st.session_state.word_frequencies)
                st.dataframe(comparison_df.head(20), use_container_width=True)
        
        # Tab 2: Similarity Analysis
        with tab2:
            st.header("Document Similarity Analysis")
            st.markdown("Compare documents using TF-IDF and BERT semantic embeddings")
            st.markdown("---")
            
            tfidf_sim = st.session_state.tfidf_similarity
            bert_sim = st.session_state.bert_similarity
            
            if tfidf_sim is not None and bert_sim is not None:
                # Display TF-IDF
                display_similarity_analysis(tfidf_sim, "TF-IDF")
                st.markdown("---")
                
                # Display BERT
                display_similarity_analysis(bert_sim, "BERT")
                st.markdown("---")
                
                # Comparison
                compare_similarity_methods(tfidf_sim, bert_sim)
            else:
                st.warning("⚠️ Similarity analysis requires at least 2 documents")
        
        # Tab 3: Sentiment Analysis
        with tab3:
            st.header("Sentiment Analysis")
            st.markdown("Analyze the emotional tone of regulatory documents using VADER")
            st.markdown("---")
            
            sentiment_results = get_all_sentiment_results_list()
            
            if sentiment_results:
                # Display comparison first
                if len(sentiment_results) > 1:
                    display_sentiment_comparison(sentiment_results)
                    st.markdown("---")
                    
                    # Display summary
                    summary = get_sentiment_summary(sentiment_results)
                    st.markdown(summary)
                    st.markdown("---")
                
                # Display individual results
                st.subheader("📄 Individual Document Analysis")
                for result in sentiment_results:
                    with st.expander(f"View details: {result['doc_name']}", expanded=len(sentiment_results)==1):
                        display_sentiment_analysis(result)
            else:
                st.warning("⚠️ No sentiment analysis results available")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Streamlit • Powered by spaCy, BERT & VADER"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
