from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def make_tools(session: Any) -> List[Any]:
    """Factory function creating DTM tools bound to session."""

    @tool
    def build_dtm_tool(
        vectorizer_type: str = "count",
        ngram_range: List[int] = [1, 1],
        min_df: float = 1,
        max_df: float = 1.0,
        stop_words: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build a Document-Term Matrix (DTM) from session.dataframe["text"].

        Args:
            vectorizer_type: 'count' for CountVectorizer or 'tfidf' for TfidfVectorizer.
            ngram_range: List of two integers [min_n, max_n], e.g. [1, 1] for unigrams.
            min_df: Minimum document frequency threshold (int for count, float for ratio).
            max_df: Maximum document frequency threshold (float for ratio, int for count).
            stop_words: Optional language string for stop words removal (e.g. 'english').
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}
        if "text" not in session.dataframe.columns:
            return {"error": "Column 'text' not found in dataframe."}

        # Handle empty/NaN text entries gracefully by replacing with empty string
        texts = session.dataframe["text"].fillna("").astype(str).tolist()

        # Convert List[int] to Tuple[int, int] for scikit-learn
        ngram_tuple = (ngram_range[0], ngram_range[1])

        if vectorizer_type.lower() == "tfidf":
            vectorizer = TfidfVectorizer(
                ngram_range=ngram_tuple,
                min_df=min_df,
                max_df=max_df,
                stop_words=stop_words
            )
        else:
            vectorizer = CountVectorizer(
                ngram_range=ngram_tuple,
                min_df=min_df,
                max_df=max_df,
                stop_words=stop_words
            )

        X = vectorizer.fit_transform(texts)
        feature_names = list(vectorizer.get_feature_names_out())

        # Store vectorizer and feature matrix in session
        session.artifacts["count_vectorizer"] = vectorizer
        session.feature_matrix = X
        session.feature_names = feature_names

        # Compute sparsity stats
        non_zero = int(X.nnz)
        n_rows, n_cols = X.shape
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - (non_zero / total_elements))) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "vectorizer_type": vectorizer_type,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4)
        }

        session.store_result("build_dtm_tool", {"vectorizer_type": vectorizer_type, "ngram_range": ngram_range}, summary)
        return summary

    @tool
    def term_frequency_tool(
        top_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """Aggregate total term frequencies across all documents in session.feature_matrix.

        Args:
            top_n: Optional integer specifying how many top terms to return (sorted by frequency).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix or feature names found in session. Run build_dtm_tool first."}

        X = session.feature_matrix
        feature_names = session.feature_names

        # Sum total term frequency across documents (columns sum)
        if issparse(X):
            frequencies = np.array(X.sum(axis=0)).ravel()
        else:
            frequencies = np.sum(np.asarray(X), axis=0)

        # Create term to frequency dictionary
        term_freq_tuples = [(term, int(freq)) for term, freq in zip(feature_names, frequencies)]
        
        # Sort descending by frequency
        term_freq_tuples.sort(key=lambda x: x[1], reverse=True)

        if top_n is not None and top_n > 0:
            term_freq_tuples = term_freq_tuples[:top_n]

        term_frequencies = {term: freq for term, freq in term_freq_tuples}

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "total_terms": len(feature_names),
            "returned_terms": len(term_frequencies),
            "term_frequencies": term_frequencies
        }

        session.store_result("term_frequency_tool", {"top_n": top_n}, summary)
        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20
    ) -> Dict[str, Any]:
        """Render a heatmap visualization of a raw positional slice of the Document-Term Matrix.

        Args:
            n_terms: Number of terms (columns) to include from the start of session.feature_names.
            n_documents: Number of documents (rows) to include from the start of session.dataframe.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix or feature names found in session. Run build_dtm_tool first."}

        X = session.feature_matrix
        feature_names = session.feature_names

        sliced_terms = feature_names[:n_terms]
        X_slice = X[:n_documents, :n_terms]

        dense_slice = X_slice.toarray() if issparse(X_slice) else np.asarray(X_slice)
        n_docs_sliced, n_terms_sliced = dense_slice.shape

        doc_labels = [f"Doc {i}" for i in range(n_docs_sliced)]

        # Create figure and plot heatmap
        fig, ax = plt.subplots(figsize=(max(6, n_terms_sliced * 0.5), max(4, n_docs_sliced * 0.4)))
        sns.heatmap(
            dense_slice,
            annot=True,
            fmt="g",
            xticklabels=sliced_terms,
            yticklabels=doc_labels,
            cmap="Blues",
            cbar=True,
            ax=ax
        )
        ax.set_title("Document-Term Matrix Heatmap")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Set pending figure on session
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents": n_docs_sliced,
            "n_terms": n_terms_sliced,
            "document_indices": list(range(n_docs_sliced)),
            "feature_names": sliced_terms,
            "matrix_slice": dense_slice.tolist()
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
