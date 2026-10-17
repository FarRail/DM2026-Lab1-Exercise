from typing import Optional, Literal, Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool
from agent_pipeline.session_state import SessionState

def make_tools(session: SessionState) -> List[Any]:
    """Factory function creating tools for dimensionality reduction and label binarization."""

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = 42
    ) -> Dict[str, Any]:
        """Reduces document-term matrix dimensionality (PCA, t-SNE, UMAP) and creates a 2D scatter plot.

        Args:
            method: Reduction technique ('pca', 'tsne', or 'umap'). Default 'pca'.
            n_components: Target dimensions (default 2).
            perplexity: Perplexity setting for t-SNE (default 30.0).
            n_neighbors: Number of neighbors for UMAP (default 15).
            random_state: Seed for reproducibility (default 42).

        Returns:
            Dict containing result_id, method, coordinates, and explained_variance_ratio (if PCA).
        """
        if session.feature_matrix is None:
            raise ValueError("No feature_matrix found in session. Run build_dtm_tool first.")

        # Convert sparse DTM to dense numpy array
        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)

        explained_variance_ratio = None

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance_ratio = reducer.explained_variance_ratio_.tolist()
        elif method == "tsne":
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state
            )
            coords = reducer.fit_transform(X)
        elif method == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state
            )
            coords = reducer.fit_transform(X)
        else:
            raise ValueError(f"Unsupported reduction method: {method}")

        # Extract category labels for coloring the scatter plot
        labels = None
        if session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                labels = session.dataframe["category_name"].tolist()
            elif "category" in session.dataframe.columns:
                labels = session.dataframe["category"].tolist()

        if labels is None and session.labels is not None:
            labels = list(session.labels)
        if labels is None:
            labels = ["unlabeled"] * X.shape[0]

        # Mandatory Plotting Requirement: Create 2D scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_df = pd.DataFrame({
            "Dim 1": coords[:, 0],
            "Dim 2": coords[:, 1] if n_components > 1 else np.zeros(coords.shape[0]),
            "Category": labels
        })

        sns.scatterplot(
            data=plot_df,
            x="Dim 1",
            y="Dim 2",
            hue="Category",
            style="Category",
            s=100,
            ax=ax
        )
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Set figure on session for notebook display
        session.pending_figure = fig

        coords_list = coords.tolist()
        session.artifacts["reduced_coordinates"] = coords_list

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "n_documents": X.shape[0],
            "coordinates": coords_list
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result("reduce_dimensions_tool", {"method": method}, summary)
        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encodes category labels in session.dataframe."""
        if session.dataframe is None:
            raise ValueError("No dataframe found in session.")

        if "category_name" in session.dataframe.columns:
            target_col = "category_name"
        elif "category" in session.dataframe.columns:
            target_col = "category"
        else:
            raise ValueError("No category column found in session.dataframe.")

        one_hot = pd.get_dummies(session.dataframe[target_col], prefix="cat", dtype=int)
        for col in one_hot.columns:
            session.dataframe[col] = one_hot[col]

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "binarized_columns": list(one_hot.columns),
            "n_rows": len(session.dataframe)
        }
        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
