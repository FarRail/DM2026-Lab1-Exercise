from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool
from agent_pipeline.session_state import SessionState

def make_tools(session: SessionState) -> List[Any]:
    """Factory function creating tools for data exploration (cosine similarity & feature correlation)."""

    @tool
    def cosine_similarity_tool(
        doc_index_1: int,
        doc_index_2: int
    ) -> Dict[str, Any]:
        """Calculates cosine similarity between two documents in the document-term matrix.

        Args:
            doc_index_1: Index of the first document.
            doc_index_2: Index of the second document.

        Returns:
            Dict containing result_id, doc_index_1, doc_index_2, and cosine_similarity float.
        """
        if session.feature_matrix is None:
            raise ValueError("No feature_matrix found in session. Run build_dtm_tool first.")

        n_docs = session.feature_matrix.shape[0]
        if not (0 <= doc_index_1 < n_docs and 0 <= doc_index_2 < n_docs):
            raise IndexError(f"Document indices must be between 0 and {n_docs - 1}.")

        # Extract rows from sparse feature matrix
        vec1 = session.feature_matrix[doc_index_1].toarray() if hasattr(session.feature_matrix[doc_index_1], "toarray") else session.feature_matrix[doc_index_1]
        vec2 = session.feature_matrix[doc_index_2].toarray() if hasattr(session.feature_matrix[doc_index_2], "toarray") else session.feature_matrix[doc_index_2]

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # Edge case handling for empty vectors (e.g. empty text rows)
        if norm1 == 0 or norm2 == 0:
            sim_score = 0.0
        else:
            sim_score = float(cosine_similarity(vec1, vec2)[0, 0])

        # Fetch document text if available
        text_1, text_2 = None, None
        if session.dataframe is not None and "text" in session.dataframe.columns:
            text_1 = str(session.dataframe.iloc[doc_index_1]["text"])
            text_2 = str(session.dataframe.iloc[doc_index_2]["text"])

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "cosine_similarity": round(sim_score, 4)
        }
        if text_1 is not None:
            summary["text_1"] = text_1
            summary["text_2"] = text_2

        session.store_result("cosine_similarity_tool", {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2}, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool() -> Dict[str, Any]:
        """Computes feature-vs-feature correlation matrix for top 20 terms by variance and generates a heatmap.

        Returns:
            Dict containing result_id, terms, and feature correlation matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No feature_matrix or feature_names found in session. Run build_dtm_tool first.")

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
        feature_names = list(session.feature_names)

        # Compute variance per term across all documents
        variances = np.var(X, axis=0)

        # Select top 20 terms by variance
        top_k = min(20, X.shape[1])
        top_indices = np.argsort(variances)[::-1][:top_k]

        top_X = X[:, top_indices]
        top_terms = [feature_names[i] for i in top_indices]

        # Compute Pearson correlation matrix across top terms
        corr_matrix = np.corrcoef(top_X, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Plot annotated correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            xticklabels=top_terms,
            yticklabels=top_terms,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax
        )
        ax.set_title("Feature Correlation Matrix (Top Terms by Variance)")
        plt.tight_layout()

        # Mandatory Plotting Requirement: Set figure on session
        session.pending_figure = fig

        result_id = session.next_result_id("feature_correlation")
        summary = {
            "result_id": result_id,
            "terms": top_terms,
            "correlation_matrix": np.round(corr_matrix, 4).tolist()
        }

        session.store_result("feature_correlation_matrix_tool", {}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
