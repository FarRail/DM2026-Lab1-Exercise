from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import spearmanr

def make_tools(session: Any) -> List[Any]:
    """Factory function creating feature filtering tools bound to session."""

    @tool
    def variance_filter_tool(
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Filter terms from session.feature_matrix whose variance is below a threshold.

        Args:
            threshold: Minimum variance threshold. Features with variance < threshold are removed.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix or feature names found in session. Run build_dtm_tool first."}

        X = session.feature_matrix
        feature_names = np.array(session.feature_names)

        # Compute mean and mean square per column without densifying
        if issparse(X):
            mean = np.array(X.mean(axis=0)).ravel()
            mean_sq = np.array(X.power(2).mean(axis=0)).ravel()
            variances = mean_sq - (mean ** 2)
        else:
            variances = np.var(X, axis=0)

        # Identify kept and removed feature indices
        kept_mask = variances >= threshold
        removed_mask = ~kept_mask

        kept_indices = np.where(kept_mask)[0]
        removed_indices = np.where(removed_mask)[0]

        kept_names = feature_names[kept_indices].tolist()
        removed_names = feature_names[removed_indices].tolist()

        # Build full_report sorted descending by variance for visualize_result_tool
        report_df = pd.DataFrame({
            "term": feature_names,
            "variance": np.round(variances, 6)
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Update session state with filtered feature matrix and feature names
        session.feature_matrix = X[:, kept_indices]
        session.feature_names = kept_names

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "initial_features": len(feature_names),
            "kept_features": len(kept_names),
            "removed_features": len(removed_names),
            "kept_feature_names": kept_names,
            "removed_feature_names": removed_names
        }

        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=report_df)
        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Filter terms from session.feature_matrix based on Pearson correlation with target_class.

        Args:
            target_class: Category label string to correlate features against (one-vs-rest).
            threshold: Absolute correlation threshold. Features with |pearson_r| < threshold are removed.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix or feature names found in session. Run build_dtm_tool first."}
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"error": "No dataframe or 'category_name' column found in session."}

        X = session.feature_matrix
        feature_names = np.array(session.feature_names)

        # Create binary target vector y
        y = (session.dataframe["category_name"].values == target_class).astype(float)
        if np.std(y) == 0:
            return {"error": f"Target class '{target_class}' not found or present in all documents."}

        # Dense array for feature matrix X
        X_dense = X.toarray() if issparse(X) else np.asarray(X)

        # Calculate Pearson r vectorized
        y_centered = y - np.mean(y)
        y_std = np.std(y)

        X_centered = X_dense - np.mean(X_dense, axis=0)
        X_std = np.std(X_dense, axis=0)

        # Avoid division by zero for constant terms
        with np.errstate(divide='ignore', invalid='ignore'):
            correlations = np.mean(X_centered * y_centered[:, None], axis=0) / (X_std * y_std)
            correlations = np.nan_to_num(correlations, nan=0.0)

        # Identify kept and removed features based on absolute threshold
        abs_correlations = np.abs(correlations)
        kept_mask = abs_correlations >= threshold
        removed_mask = ~kept_mask

        kept_indices = np.where(kept_mask)[0]
        removed_indices = np.where(removed_mask)[0]

        kept_names = feature_names[kept_indices].tolist()
        removed_names = feature_names[removed_indices].tolist()

        # Build full_report sorted descending by pearson_r for visualize_result_tool
        report_df = pd.DataFrame({
            "term": feature_names,
            "pearson_r": np.round(correlations, 4)
        }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        # Update session state with filtered feature matrix and feature names
        session.feature_matrix = X[:, kept_indices] if issparse(X) else X_dense[:, kept_indices]
        session.feature_names = kept_names

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "initial_features": len(feature_names),
            "kept_features": len(kept_names),
            "removed_features": len(removed_names),
            "kept_feature_names": kept_names,
            "removed_feature_names": removed_names,
            "correlations": {term: round(float(r), 4) for term, r in zip(feature_names, correlations)}
        }

        session.store_result("pearson_filter_tool", {"target_class": target_class, "threshold": threshold}, summary, full_report=report_df)
        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Filter terms from session.feature_matrix based on Spearman rank correlation with target_class.

        Args:
            target_class: Category label string to correlate features against (one-vs-rest).
            threshold: Absolute correlation threshold. Features with |spearman_r| < threshold are removed.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix or feature names found in session. Run build_dtm_tool first."}
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"error": "No dataframe or 'category_name' column found in session."}

        X = session.feature_matrix
        feature_names = np.array(session.feature_names)

        # Create binary target vector y
        y = (session.dataframe["category_name"].values == target_class).astype(float)
        if np.std(y) == 0:
            return {"error": f"Target class '{target_class}' not found or present in all documents."}

        # Dense array for feature matrix X
        X_dense = X.toarray() if issparse(X) else np.asarray(X)

        # Calculate Spearman correlation for each column against y
        correlations = []
        for i in range(X_dense.shape[1]):
            col = X_dense[:, i]
            if np.std(col) == 0:
                correlations.append(0.0)
            else:
                r, _ = spearmanr(col, y)
                correlations.append(0.0 if np.isnan(r) else float(r))

        correlations = np.array(correlations)

        # Identify kept and removed features based on absolute threshold
        abs_correlations = np.abs(correlations)
        kept_mask = abs_correlations >= threshold
        removed_mask = ~kept_mask

        kept_indices = np.where(kept_mask)[0]
        removed_indices = np.where(removed_mask)[0]

        kept_names = feature_names[kept_indices].tolist()
        removed_names = feature_names[removed_indices].tolist()

        # Build full_report sorted descending by spearman_r for visualize_result_tool
        report_df = pd.DataFrame({
            "term": feature_names,
            "spearman_r": np.round(correlations, 4)
        }).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        # Update session state with filtered feature matrix and feature names
        session.feature_matrix = X[:, kept_indices] if issparse(X) else X_dense[:, kept_indices]
        session.feature_names = kept_names

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "initial_features": len(feature_names),
            "kept_features": len(kept_names),
            "removed_features": len(removed_names),
            "kept_feature_names": kept_names,
            "removed_feature_names": removed_names,
            "correlations": {term: round(float(r), 4) for term, r in zip(feature_names, correlations)}
        }

        session.store_result("spearman_filter_tool", {"target_class": target_class, "threshold": threshold}, summary, full_report=report_df)
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
