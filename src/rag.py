"""
Agentic HRMS — RAG Pipeline for HR Policy Retrieval
TF-IDF based document retrieval with chunking and similarity search.
"""
import streamlit as st
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICIES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "policies")


def load_policy_documents():
    """Load all policy documents from the policies directory."""
    documents = []
    if not os.path.exists(POLICIES_DIR):
        return documents

    for filename in os.listdir(POLICIES_DIR):
        if filename.endswith((".md", ".txt")):
            filepath = os.path.join(POLICIES_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            doc_name = filename.replace("_", " ").replace(".md", "").replace(".txt", "").title()
            documents.append({
                "name": doc_name,
                "filename": filename,
                "content": content,
            })
    return documents


def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks for better retrieval."""
    # Split by sections first (headers)
    sections = re.split(r'\n##?\s+', text)
    chunks = []

    for section in sections:
        section = section.strip()
        if len(section) <= chunk_size:
            if section:
                chunks.append(section)
        else:
            # Split long sections into overlapping chunks
            words = section.split()
            current_chunk = []
            current_len = 0

            for word in words:
                current_chunk.append(word)
                current_len += len(word) + 1

                if current_len >= chunk_size:
                    chunks.append(" ".join(current_chunk))
                    # Keep overlap
                    overlap_words = int(overlap / 5)  # ~5 chars per word
                    current_chunk = current_chunk[-overlap_words:]
                    current_len = sum(len(w) + 1 for w in current_chunk)

            if current_chunk:
                chunks.append(" ".join(current_chunk))

    return chunks


@st.cache_resource(show_spinner="Building knowledge base...")
def build_rag_index():
    """Build the RAG index from policy documents."""
    documents = load_policy_documents()
    if not documents:
        return None

    all_chunks = []
    chunk_metadata = []

    for doc in documents:
        chunks = chunk_text(doc["content"])
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_metadata.append({
                "source": doc["name"],
                "filename": doc["filename"],
            })

    if not all_chunks:
        return None

    # Build TF-IDF index
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(all_chunks)

    return {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "chunks": all_chunks,
        "metadata": chunk_metadata,
        "documents": documents,
    }


def search_policies(query, rag_index, top_k=3):
    """
    Search policy documents for relevant information.
    Returns list of (chunk, source, score) tuples.
    """
    if rag_index is None:
        return []

    query_vec = rag_index["vectorizer"].transform([query])
    similarities = cosine_similarity(query_vec, rag_index["tfidf_matrix"])[0]

    # Get top-k results
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score > 0.05:  # Minimum relevance threshold
            results.append({
                "chunk": rag_index["chunks"][idx],
                "source": rag_index["metadata"][idx]["source"],
                "score": round(float(score), 4),
            })

    return results


def format_rag_response(query, results):
    """Format RAG results into a readable response."""
    if not results:
        return "I couldn't find specific policy information related to your question. Please consult with HR directly for detailed policy guidance."

    response_parts = []
    response_parts.append(f"Based on our HR policy documents, here's what I found:\n")

    sources_used = set()
    for result in results:
        sources_used.add(result["source"])
        # Clean up the chunk for display
        chunk = result["chunk"].strip()
        if len(chunk) > 600:
            chunk = chunk[:600] + "..."
        response_parts.append(f"**{result['source']}:**\n{chunk}\n")

    response_parts.append(f"\n📄 *Sources: {', '.join(sources_used)}*")
    response_parts.append("\n⚠️ *This is retrieved from sample policy documents for demonstration purposes. Always refer to official company policies for authoritative information.*")

    return "\n".join(response_parts)
