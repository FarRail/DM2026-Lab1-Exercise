from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize

# Ensure NLTK punkt tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        nltk.download("punkt", quiet=True)

def make_tools(session: Any) -> List[Any]:
    """Factory function creating data tools bound to session."""

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenize each document's text in session.dataframe["text"] into unigram tokens.

        Applies standard lowercase conversion and word tokenization (nltk.word_tokenize)
        to each document string, storing the resulting token lists in session.dataframe["unigrams"].
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Load a dataset first."}
        if "text" not in session.dataframe.columns:
            return {"error": "Column 'text' not found in dataframe."}

        texts = session.dataframe["text"].fillna("").astype(str)

        # Tokenize each text into lowercased unigram tokens
        tokenized_docs: List[List[str]] = []
        for text in texts:
            if not text.strip():
                tokenized_docs.append([])
            else:
                tokens = word_tokenize(text.lower())
                tokenized_docs.append(tokens)

        # Store in dataframe and artifacts
        session.dataframe["unigrams"] = tokenized_docs
        session.artifacts["unigrams"] = tokenized_docs

        total_tokens = sum(len(doc) for doc in tokenized_docs)
        n_docs = len(tokenized_docs)
        avg_tokens = round(total_tokens / n_docs, 4) if n_docs > 0 else 0.0

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": n_docs,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": avg_tokens
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def list_files_tool(
        subdirectory: str = ""
    ) -> Dict[str, Any]:
        """List files and directories in the workspace to locate datasets.

        Args:
            subdirectory: Relative subdirectory path from workspace root to list (default is root "").
        """
        target_path = pathlib.Path(subdirectory)
        if not target_path.exists():
            return {"error": f"Path '{subdirectory}' does not exist."}
        if not target_path.is_dir():
            return {"error": f"Path '{subdirectory}' is not a directory."}

        directories: List[str] = []
        files: List[str] = []

        for entry in os.scandir(target_path):
            name = entry.name
            # Skip hidden files/directories and __pycache__
            if name.startswith(".") or name == "__pycache__":
                continue
            if entry.is_dir():
                directories.append(name)
            elif entry.is_file():
                files.append(name)

        directories.sort()
        files.sort()

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(
        n_rows: int = 5,
        mode: str = "head"
    ) -> Dict[str, Any]:
        """Peek at top ('head') or bottom ('tail') rows of session.dataframe along with column metadata.

        Args:
            n_rows: Number of rows to inspect (default 5).
            mode: Inspection direction, 'head' for first rows or 'tail' for last rows (default 'head').
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Load a dataset first."}

        df = session.dataframe

        if mode.lower() == "tail":
            inspected_df = df.tail(n_rows)
        else:
            inspected_df = df.head(n_rows)

        # Convert inspect slice to JSON-serializable list of dict records
        records = inspected_df.to_dict(orient="records")

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "inspected_rows": len(records),
            "mode": mode,
            "data": records
        }

        session.store_result("inspect_data_tool", {"n_rows": n_rows, "mode": mode}, summary)
        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Audit session.dataframe["text"] for missing, null, or empty string values."""
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Load a dataset first."}
        if "text" not in session.dataframe.columns:
            return {"error": "Column 'text' not found in dataframe."}

        df = session.dataframe
        text_col = df["text"]

        # Identify missing: NaN/None or whitespace-only strings
        missing_mask = text_col.isna() | text_col.astype(str).str.strip().eq("")
        missing_indices = df.index[missing_mask].tolist()
        missing_count = len(missing_indices)

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_rows": len(df),
            "missing_count": missing_count,
            "missing_indices": missing_indices
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(
        drop: bool = False
    ) -> Dict[str, Any]:
        """Check for (and optionally drop) duplicate documents in session.dataframe["text"].

        Args:
            drop: If True, drops duplicate text documents using keep=False (removing all duplicate copies).
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Load a dataset first."}
        if "text" not in session.dataframe.columns:
            return {"error": "Column 'text' not found in dataframe."}

        df = session.dataframe
        total_rows_before = len(df)

        # Identify duplicate rows (keep="first" counts extra occurrences)
        duplicate_mask = df["text"].duplicated(keep="first")
        duplicate_indices = df.index[duplicate_mask].tolist()
        duplicate_count = len(duplicate_indices)

        if drop and duplicate_count > 0:
            # Match Master's drop_duplicates(keep=False) requirement exactly
            session.dataframe = df.drop_duplicates(subset=["text"], keep=False).reset_index(drop=True)

        total_rows_after = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "total_rows_before": total_rows_before,
            "duplicate_count": duplicate_count,
            "duplicate_indices": duplicate_indices,
            "dropped": drop,
            "total_rows_after": total_rows_after
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """Subsample session.dataframe by exact count n or fraction frac.

        Args:
            n: Number of items to sample.
            frac: Fraction of axis items to sample.
            random_state: Seed for random number generator for reproducible sampling.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Load a dataset first."}

        df = session.dataframe
        total_rows_before = len(df)

        if n is None and frac is None:
            n = min(5, total_rows_before)

        sampled_df = df.sample(n=n, frac=frac, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        # Update session state dataframe with sampled rows and reset index
        session.dataframe = sampled_df.reset_index(drop=True)

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "total_rows_before": total_rows_before,
            "sampled_rows": len(session.dataframe),
            "sampled_indices": sampled_indices,
            "random_state": random_state
        }

        session.store_result("sample_data_tool", {"n": n, "frac": frac, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Compute text length statistics (overall and per category) and generate a box plot."""
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Load a dataset first."}
        if "text" not in session.dataframe.columns:
            return {"error": "Column 'text' not found in dataframe."}

        df = session.dataframe

        # Character length per document matching Master's df["text"].apply(len)
        text_lengths = df["text"].fillna("").astype(str).apply(len)
        df["text_length"] = text_lengths

        overall_stats = {k: float(np.round(v, 4)) for k, v in text_lengths.describe().to_dict().items()}

        per_category_stats: Dict[str, Dict[str, float]] = {}
        if "category_name" in df.columns:
            grouped = df.groupby("category_name")["text_length"].describe()
            for cat, row in grouped.iterrows():
                per_category_stats[str(cat)] = {k: float(np.round(v, 4)) for k, v in row.to_dict().items()}

        # Generate box plot grouped by category_name
        fig, ax = plt.subplots(figsize=(6, 4))
        if "category_name" in df.columns:
            sns.boxplot(x="category_name", y="text_length", data=df, ax=ax)
            ax.set_title("Document Text Length by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(y="text_length", data=df, ax=ax)
            ax.set_title("Document Text Length Distribution")
        ax.set_ylabel("Text Length (characters)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "n_documents": len(df),
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    return [
        tokenize_tool,
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool
    ]
