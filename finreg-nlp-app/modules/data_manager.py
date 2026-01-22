"""
Data Manager Module
Handles session state management, caching, and data validation
"""

import streamlit as st
from utils.config import MAX_FILE_UPLOADS


def initialize_session_state():
    """
    Initialize all session state variables.
    Should be called at the start of the app.
    """
    # Uploaded files tracking
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    # Processed documents data
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = {}
    
    # Raw text data (for sentiment analysis)
    if 'raw_texts' not in st.session_state:
        st.session_state.raw_texts = {}
    
    # Cleaned text data (for other analyses)
    if 'cleaned_texts' not in st.session_state:
        st.session_state.cleaned_texts = {}
    
    # Word frequencies
    if 'word_frequencies' not in st.session_state:
        st.session_state.word_frequencies = {}
    
    # Similarity results
    if 'tfidf_similarity' not in st.session_state:
        st.session_state.tfidf_similarity = None
    
    if 'bert_similarity' not in st.session_state:
        st.session_state.bert_similarity = None
    
    # Sentiment results
    if 'sentiment_results' not in st.session_state:
        st.session_state.sentiment_results = {}
    
    # Processing status
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    # Models loaded flag
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False


def cache_processed_data(key, value):
    """
    Store data in session state.
    
    Args:
        key: Session state key
        value: Data to store
    """
    st.session_state[key] = value


def get_cached_data(key, default=None):
    """
    Retrieve data from session state.
    
    Args:
        key: Session state key
        default: Default value if key doesn't exist
    
    Returns:
        Cached data or default value
    """
    return st.session_state.get(key, default)


def clear_all_cache():
    """
    Clear all cached data and reset session state.
    """
    keys_to_clear = [
        'uploaded_files',
        'processed_docs',
        'raw_texts',
        'cleaned_texts',
        'word_frequencies',
        'tfidf_similarity',
        'bert_similarity',
        'sentiment_results',
        'processing_complete'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # Reinitialize
    initialize_session_state()


def validate_uploaded_files(uploaded_files):
    """
    Validate uploaded files.
    
    Args:
        uploaded_files: List of UploadedFile objects
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not uploaded_files:
        return False, "⚠️ Please upload at least one PDF file."
    
    if len(uploaded_files) > MAX_FILE_UPLOADS:
        return False, f"⚠️ Maximum {MAX_FILE_UPLOADS} files allowed. You uploaded {len(uploaded_files)}."
    
    # Check file types
    for file in uploaded_files:
        if not file.name.lower().endswith('.pdf'):
            return False, f"⚠️ Invalid file type: {file.name}. Only PDF files are allowed."
    
    # Check for duplicate names
    file_names = [f.name for f in uploaded_files]
    if len(file_names) != len(set(file_names)):
        return False, "⚠️ Duplicate file names detected. Please upload files with unique names."
    
    return True, ""


def get_document_names():
    """
    Get list of processed document names.
    
    Returns:
        list: Document names
    """
    return list(st.session_state.cleaned_texts.keys())


def has_processed_documents():
    """
    Check if any documents have been processed.
    
    Returns:
        bool: True if documents are processed
    """
    return len(st.session_state.cleaned_texts) > 0


def get_document_count():
    """
    Get the number of processed documents.
    
    Returns:
        int: Number of documents
    """
    return len(st.session_state.cleaned_texts)


def is_processing_complete():
    """
    Check if processing is complete.
    
    Returns:
        bool: True if processing is complete
    """
    return st.session_state.get('processing_complete', False)


def set_processing_complete(status=True):
    """
    Set processing complete status.
    
    Args:
        status: Processing status (True/False)
    """
    st.session_state.processing_complete = status


def store_document_data(doc_name, raw_text, cleaned_text, word_freq):
    """
    Store all data for a processed document.
    
    Args:
        doc_name: Document name
        raw_text: Raw extracted text
        cleaned_text: Cleaned text
        word_freq: Word frequency series
    """
    st.session_state.raw_texts[doc_name] = raw_text
    st.session_state.cleaned_texts[doc_name] = cleaned_text
    st.session_state.word_frequencies[doc_name] = word_freq


def get_document_data(doc_name):
    """
    Retrieve all data for a document.
    
    Args:
        doc_name: Document name
    
    Returns:
        dict: Document data or None if not found
    """
    if doc_name not in st.session_state.cleaned_texts:
        return None
    
    return {
        'raw_text': st.session_state.raw_texts.get(doc_name, ''),
        'cleaned_text': st.session_state.cleaned_texts.get(doc_name, ''),
        'word_freq': st.session_state.word_frequencies.get(doc_name, None)
    }


def store_similarity_results(tfidf_sim, bert_sim):
    """
    Store similarity analysis results.
    
    Args:
        tfidf_sim: TF-IDF similarity DataFrame
        bert_sim: BERT similarity DataFrame
    """
    st.session_state.tfidf_similarity = tfidf_sim
    st.session_state.bert_similarity = bert_sim


def get_similarity_results():
    """
    Retrieve similarity analysis results.
    
    Returns:
        tuple: (tfidf_similarity, bert_similarity)
    """
    return (
        st.session_state.get('tfidf_similarity'),
        st.session_state.get('bert_similarity')
    )


def store_sentiment_result(doc_name, sentiment_data):
    """
    Store sentiment analysis result for a document.
    
    Args:
        doc_name: Document name
        sentiment_data: Sentiment analysis result dictionary
    """
    st.session_state.sentiment_results[doc_name] = sentiment_data


def get_sentiment_results():
    """
    Retrieve all sentiment analysis results.
    
    Returns:
        dict: Sentiment results for all documents
    """
    return st.session_state.get('sentiment_results', {})


def get_all_sentiment_results_list():
    """
    Get sentiment results as a list (for visualization).
    
    Returns:
        list: List of sentiment result dictionaries
    """
    return list(st.session_state.sentiment_results.values())


def export_session_summary():
    """
    Export a summary of the current session.
    
    Returns:
        dict: Session summary
    """
    return {
        'num_documents': get_document_count(),
        'document_names': get_document_names(),
        'processing_complete': is_processing_complete(),
        'has_tfidf': st.session_state.tfidf_similarity is not None,
        'has_bert': st.session_state.bert_similarity is not None,
        'has_sentiment': len(st.session_state.sentiment_results) > 0
    }
