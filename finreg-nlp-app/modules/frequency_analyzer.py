"""
Frequency Analysis Module
Handles word frequency analysis and WordCloud generation
"""

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud
from utils.config import (
    WORDCLOUD_WIDTH,
    WORDCLOUD_HEIGHT,
    WORDCLOUD_MAX_WORDS,
    WORDCLOUD_BACKGROUND,
    TOP_N_WORDS
)


def gen_freq(tokens):
    """
    Generate word frequency distribution from tokens.
    
    Args:
        tokens: spaCy Doc object or list of tokens
    
    Returns:
        pd.Series: Word frequency series (sorted in descending order)
    """
    try:
        # Convert spaCy tokens to strings
        if hasattr(tokens, '__iter__') and hasattr(tokens[0], 'text'):
            words = [token.text for token in tokens]
        else:
            words = tokens
        
        # Create pandas Series with word frequencies
        word_freq = pd.Series(words).value_counts()
        
        return word_freq
    
    except Exception as e:
        st.error(f"❌ Error generating word frequencies: {str(e)}")
        return pd.Series()


def create_word_df(word_freq, top_n=TOP_N_WORDS):
    """
    Create a DataFrame of top N words with their frequencies.
    
    Args:
        word_freq: pandas Series with word frequencies
        top_n: Number of top words to include
    
    Returns:
        pd.DataFrame: DataFrame with 'word' and 'count' columns
    """
    try:
        # Get top N words
        top_words = word_freq.head(top_n)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Word': top_words.index,
            'Frequency': top_words.values
        })
        
        # Reset index
        df.reset_index(drop=True, inplace=True)
        df.index = df.index + 1  # Start index from 1
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error creating word DataFrame: {str(e)}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def generate_wordcloud(word_freq_dict, width=WORDCLOUD_WIDTH, height=WORDCLOUD_HEIGHT, 
                       max_words=WORDCLOUD_MAX_WORDS, background_color=WORDCLOUD_BACKGROUND):
    """
    Generate WordCloud visualization from word frequencies.
    
    Args:
        word_freq_dict: Dictionary or Series of word frequencies
        width: WordCloud width
        height: WordCloud height
        max_words: Maximum number of words to display
        background_color: Background color
    
    Returns:
        matplotlib.figure.Figure: WordCloud figure
    """
    try:
        # Convert Series to dict if needed
        if isinstance(word_freq_dict, pd.Series):
            word_freq_dict = word_freq_dict.to_dict()
        
        # Generate WordCloud
        wc = WordCloud(
            width=width,
            height=height,
            max_words=max_words,
            background_color=background_color,
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(word_freq_dict)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        return fig
    
    except Exception as e:
        st.error(f"❌ Error generating WordCloud: {str(e)}")
        return None


def display_frequency_analysis(doc_name, word_freq):
    """
    Display comprehensive frequency analysis for a document.
    
    Args:
        doc_name: Name of the document
        word_freq: pandas Series with word frequencies
    """
    st.subheader(f"📈 {doc_name}")
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Words", f"{word_freq.sum():,}")
    with col2:
        st.metric("Unique Words", f"{len(word_freq):,}")
    with col3:
        if len(word_freq) > 0:
            top_word = word_freq.index[0]
            top_count = word_freq.values[0]
            st.metric("Most Frequent", f"{top_word} ({top_count})")
    
    # Create two columns for table and wordcloud
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"**Top {TOP_N_WORDS} Words**")
        df = create_word_df(word_freq, TOP_N_WORDS)
        st.dataframe(df, use_container_width=True, height=400)
    
    with col2:
        st.markdown("**Word Cloud**")
        fig = generate_wordcloud(word_freq)
        if fig:
            st.pyplot(fig)
            plt.close()


def get_top_n_words(word_freq, n=TOP_N_WORDS):
    """
    Get top N words and their frequencies.
    
    Args:
        word_freq: pandas Series with word frequencies
        n: Number of top words to return
    
    Returns:
        list: List of tuples (word, frequency)
    """
    top_words = word_freq.head(n)
    return list(zip(top_words.index, top_words.values))


def compare_word_frequencies(freq_dict):
    """
    Compare word frequencies across multiple documents.
    
    Args:
        freq_dict: Dictionary of {doc_name: word_freq_series}
    
    Returns:
        pd.DataFrame: Comparison DataFrame
    """
    try:
        # Get top words from all documents
        all_top_words = set()
        for doc_name, freq in freq_dict.items():
            top_words = freq.head(TOP_N_WORDS).index.tolist()
            all_top_words.update(top_words)
        
        # Create comparison DataFrame
        comparison_data = {}
        for doc_name, freq in freq_dict.items():
            comparison_data[doc_name] = [freq.get(word, 0) for word in all_top_words]
        
        df = pd.DataFrame(comparison_data, index=list(all_top_words))
        df = df.loc[df.sum(axis=1).sort_values(ascending=False).index]
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error comparing frequencies: {str(e)}")
        return pd.DataFrame()


def export_frequency_data(word_freq, filename="word_frequencies.csv"):
    """
    Export word frequencies to CSV format (for download).
    
    Args:
        word_freq: pandas Series with word frequencies
        filename: Output filename
    
    Returns:
        str: CSV data as string
    """
    df = pd.DataFrame({
        'Word': word_freq.index,
        'Frequency': word_freq.values
    })
    return df.to_csv(index=False)
