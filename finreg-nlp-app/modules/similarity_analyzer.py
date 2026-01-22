"""
Similarity Analysis Module
Handles TF-IDF and BERT-based similarity computation with visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from utils.config import BERT_MODEL, HEATMAP_COLORMAP


@st.cache_resource
def load_bert_model():
    """
    Load and cache the BERT model for sentence embeddings.
    
    Returns:
        SentenceTransformer: Loaded BERT model
    """
    try:
        model = SentenceTransformer(BERT_MODEL)
        return model
    except Exception as e:
        st.error(f"❌ Error loading BERT model '{BERT_MODEL}': {str(e)}")
        st.stop()


@st.cache_data(show_spinner=False)
def compute_tfidf_similarity(docs_dict):
    """
    Compute TF-IDF vectors and cosine similarity matrix.
    
    Args:
        docs_dict: Dictionary of {doc_name: cleaned_text}
    
    Returns:
        tuple: (similarity_df, tfidf_df, vectorizer)
    """
    try:
        # Extract document names and texts
        doc_names = list(docs_dict.keys())
        documents = list(docs_dict.values())
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=1000, min_df=1)
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Create TF-IDF DataFrame (optional, for inspection)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=doc_names,
            columns=vectorizer.get_feature_names_out()
        )
        
        # Compute cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix)
        
        # Create similarity DataFrame
        similarity_df = pd.DataFrame(
            cosine_sim,
            index=doc_names,
            columns=doc_names
        )
        
        return similarity_df, tfidf_df, vectorizer
    
    except Exception as e:
        st.error(f"❌ Error computing TF-IDF similarity: {str(e)}")
        return None, None, None


@st.cache_data(show_spinner=False)
def compute_bert_similarity(_model, docs_dict):
    """
    Compute BERT embeddings and cosine similarity matrix.
    Note: _model parameter has underscore prefix to prevent hashing by Streamlit
    
    Args:
        _model: Loaded SentenceTransformer model
        docs_dict: Dictionary of {doc_name: cleaned_text}
    
    Returns:
        tuple: (similarity_df, embeddings_dict)
    """
    try:
        # Extract document names and texts
        doc_names = list(docs_dict.keys())
        documents = list(docs_dict.values())
        
        # Generate embeddings
        embeddings_dict = {}
        for name, text in docs_dict.items():
            embedding = _model.encode(text)
            embeddings_dict[name] = embedding
        
        # Compute cosine similarity matrix
        n = len(doc_names)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Compute cosine similarity (1 - cosine distance)
                    sim = 1 - cosine(embeddings_dict[doc_names[i]], 
                                    embeddings_dict[doc_names[j]])
                    similarity_matrix[i, j] = sim
        
        # Create similarity DataFrame
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=doc_names,
            columns=doc_names
        )
        
        return similarity_df, embeddings_dict
    
    except Exception as e:
        st.error(f"❌ Error computing BERT similarity: {str(e)}")
        return None, None


def create_similarity_heatmap(similarity_df, title="Similarity Heatmap"):
    """
    Create a heatmap visualization of similarity matrix.
    
    Args:
        similarity_df: DataFrame with similarity scores
        title: Title for the heatmap
    
    Returns:
        matplotlib.figure.Figure: Heatmap figure
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            similarity_df,
            annot=True,
            fmt='.3f',
            cmap=HEATMAP_COLORMAP,
            square=True,
            cbar_kws={'label': 'Similarity Score'},
            vmin=0,
            vmax=1,
            ax=ax,
            linewidths=0.5,
            linecolor='gray'
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        st.error(f"❌ Error creating heatmap: {str(e)}")
        return None


def display_similarity_analysis(similarity_df, method="TF-IDF"):
    """
    Display similarity analysis results.
    
    Args:
        similarity_df: DataFrame with similarity scores
        method: Method name (TF-IDF or BERT)
    """
    st.subheader(f"🔍 {method} Similarity Analysis")
    
    # Display similarity matrix as table
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Similarity Matrix**")
        # Format the dataframe for better display
        styled_df = similarity_df.style.background_gradient(
            cmap=HEATMAP_COLORMAP,
            vmin=0,
            vmax=1
        ).format("{:.4f}")
        st.dataframe(styled_df, use_container_width=True)
    
    with col2:
        st.markdown("**Visualization**")
        fig = create_similarity_heatmap(similarity_df, f"{method} Cosine Similarity")
        if fig:
            st.pyplot(fig)
            plt.close()
    
    # Display pairwise similarities (excluding diagonal)
    st.markdown("**Pairwise Similarity Scores**")
    pairs = []
    doc_names = similarity_df.index.tolist()
    for i in range(len(doc_names)):
        for j in range(i + 1, len(doc_names)):
            pairs.append({
                'Document Pair': f"{doc_names[i]} ↔ {doc_names[j]}",
                'Similarity': similarity_df.iloc[i, j]
            })
    
    if pairs:
        pairs_df = pd.DataFrame(pairs)
        pairs_df = pairs_df.sort_values('Similarity', ascending=False)
        pairs_df['Similarity'] = pairs_df['Similarity'].apply(lambda x: f"{x:.4f}")
        st.dataframe(pairs_df, use_container_width=True, hide_index=True)


def compare_similarity_methods(tfidf_sim_df, bert_sim_df):
    """
    Compare TF-IDF and BERT similarity results side by side.
    
    Args:
        tfidf_sim_df: TF-IDF similarity DataFrame
        bert_sim_df: BERT similarity DataFrame
    """
    st.subheader("⚖️ Comparison: TF-IDF vs BERT")
    
    # Create side-by-side comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**TF-IDF Similarity**")
        fig1 = create_similarity_heatmap(tfidf_sim_df, "TF-IDF")
        if fig1:
            st.pyplot(fig1)
            plt.close()
    
    with col2:
        st.markdown("**BERT Similarity**")
        fig2 = create_similarity_heatmap(bert_sim_df, "BERT")
        if fig2:
            st.pyplot(fig2)
            plt.close()
    
    # Calculate differences
    st.markdown("**Difference Analysis (BERT - TF-IDF)**")
    diff_df = bert_sim_df - tfidf_sim_df
    
    fig_diff, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        diff_df,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        square=True,
        cbar_kws={'label': 'Difference'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    ax.set_title("Similarity Difference (BERT - TF-IDF)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_diff)
    plt.close()


def export_similarity_data(similarity_df, filename="similarity_matrix.csv"):
    """
    Export similarity matrix to CSV format.
    
    Args:
        similarity_df: Similarity DataFrame
        filename: Output filename
    
    Returns:
        str: CSV data as string
    """
    return similarity_df.to_csv()


def get_most_similar_pairs(similarity_df, n=3):
    """
    Get the N most similar document pairs.
    
    Args:
        similarity_df: Similarity DataFrame
        n: Number of pairs to return
    
    Returns:
        list: List of tuples (doc1, doc2, similarity)
    """
    pairs = []
    doc_names = similarity_df.index.tolist()
    
    for i in range(len(doc_names)):
        for j in range(i + 1, len(doc_names)):
            pairs.append((doc_names[i], doc_names[j], similarity_df.iloc[i, j]))
    
    # Sort by similarity (descending)
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    return pairs[:n]
