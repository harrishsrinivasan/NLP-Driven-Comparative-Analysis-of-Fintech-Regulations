"""
PDF Processing Module
Handles PDF text extraction, tokenization, and text cleaning
"""

import re
import io
import fitz  # PyMuPDF
import spacy
import streamlit as st
from textblob import TextBlob, Word
from nltk.corpus import stopwords
from utils.config import SPACY_MODEL, SPACY_MAX_LENGTH


@st.cache_resource
def load_spacy_model():
    """
    Load spaCy model and cache it.
    Returns the loaded spaCy NLP model.
    """
    try:
        nlp = spacy.load(SPACY_MODEL)
        nlp.max_length = SPACY_MAX_LENGTH
        return nlp
    except OSError:
        st.error(f"⚠️ spaCy model '{SPACY_MODEL}' not found. Please install it using:")
        st.code(f"python -m spacy download {SPACY_MODEL}")
        st.stop()


def extract_pdf_text(file_bytes, file_name):
    """
    Extract text from PDF file bytes using PyMuPDF.
    
    Args:
        file_bytes: PDF file content as bytes
        file_name: Name of the file (for error reporting)
    
    Returns:
        str: Extracted text from all pages
    """
    try:
        # Open PDF from bytes
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        # Extract text from all pages
        text = "\n".join([page.get_text() for page in doc])
        doc.close()
        
        if not text.strip():
            st.warning(f"⚠️ No text extracted from {file_name}. The PDF might be empty or image-based.")
            return ""
        
        return text
    
    except Exception as e:
        st.error(f"❌ Error extracting text from {file_name}: {str(e)}")
        return ""


def tokenize_text(text, nlp_model):
    """
    Tokenize text using spaCy.
    
    Args:
        text: Input text string
        nlp_model: Loaded spaCy model
    
    Returns:
        spacy.tokens.Doc: Tokenized document
    """
    try:
        doc = nlp_model(text)
        return doc
    except Exception as e:
        st.error(f"❌ Error during tokenization: {str(e)}")
        return None


@st.cache_data(show_spinner=False)
def clean_text(text):
    """
    Complete text preprocessing pipeline:
    1. Convert to lowercase
    2. Remove non-alphanumeric characters
    3. POS tagging and lemmatization
    4. Remove stopwords
    
    Args:
        text: Raw text string
    
    Returns:
        str: Cleaned and preprocessed text
    """
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove non-alphanumeric characters (keep spaces)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Use TextBlob for POS tagging and lemmatization
        blob = TextBlob(text)
        
        lemmatized_words = []
        for word, tag in blob.tags:
            # Convert TextBlob POS tag to WordNet POS tag
            if tag.startswith('J'):
                pos = 'a'  # Adjective
            elif tag.startswith('V'):
                pos = 'v'  # Verb
            elif tag.startswith('N'):
                pos = 'n'  # Noun
            elif tag.startswith('R'):
                pos = 'r'  # Adverb
            else:
                pos = 'n'  # Default to noun
            
            # Lemmatize the word
            lemmatized_word = Word(word).lemmatize(pos)
            lemmatized_words.append(lemmatized_word)
        
        # Get English stopwords
        stop_words = set(stopwords.words('english'))
        
        # Remove stopwords
        filtered_words = [word for word in lemmatized_words if word not in stop_words]
        
        # Join words back into text
        cleaned = ' '.join(filtered_words)
        
        return cleaned
    
    except Exception as e:
        st.error(f"❌ Error during text cleaning: {str(e)}")
        return text


def process_uploaded_file(uploaded_file):
    """
    Complete pipeline for processing an uploaded PDF file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        dict: {
            'file_name': str,
            'raw_text': str,
            'cleaned_text': str,
            'token_count': int,
            'success': bool
        }
    """
    result = {
        'file_name': uploaded_file.name,
        'raw_text': '',
        'cleaned_text': '',
        'token_count': 0,
        'success': False
    }
    
    try:
        # Read file bytes
        file_bytes = uploaded_file.read()
        
        # Extract text from PDF
        raw_text = extract_pdf_text(file_bytes, uploaded_file.name)
        
        if not raw_text:
            return result
        
        result['raw_text'] = raw_text
        
        # Clean the text
        cleaned = clean_text(raw_text)
        result['cleaned_text'] = cleaned
        
        # Count tokens (simple word count)
        result['token_count'] = len(cleaned.split())
        
        result['success'] = True
        return result
    
    except Exception as e:
        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        return result


def get_text_statistics(text):
    """
    Get basic statistics about the text.
    
    Args:
        text: Input text string
    
    Returns:
        dict: Statistics including character count, word count, etc.
    """
    words = text.split()
    return {
        'characters': len(text),
        'words': len(words),
        'unique_words': len(set(words)),
        'sentences': text.count('.') + text.count('!') + text.count('?')
    }
