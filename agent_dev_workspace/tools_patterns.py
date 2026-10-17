from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import PAMI.extras.convert.DF2DB as db_converter
import PAMI.frequentPattern.basic.FPGrowth as fpgrowth
import PAMI.frequentPattern.topk.FAE as topk
import PAMI.frequentPattern.maximal.MaxFPGrowth as maxfpgrowth

def make_tools(session: Any) -> List[Any]:
    """Factory function creating pattern mining tools bound to session."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "tfidf",
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Mine frequent itemsets/patterns for a single category using PAMI algorithms.

        Args:
            category_name: Target category label string to mine.
            filtering_method: Vocabulary filter method ('variance', 'tfidf', or 'term_frequency').
            algorithm: PAMI mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support count (required for 'fpgrowth' and 'maxfpgrowth').
            k: Top k patterns count (required for 'topk').
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Load a dataset first."}
        if "category_name" not in session.dataframe.columns:
            return {"error": "Column 'category_name' not found in dataframe."}
        if "text" not in session.dataframe.columns:
            return {"error": "Column 'text' not found in dataframe."}

        # Step 1: Filter dataframe by category_name
        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        if cat_df.empty:
            return {"error": f"No documents found for category '{category_name}'."}

        texts = cat_df["text"].fillna("").astype(str).tolist()

        # Step 1 cont: Fresh category-local CountVectorizer(stop_words="english")
        vectorizer = CountVectorizer(stop_words="english")
        try:
            X = vectorizer.fit_transform(texts)
            feature_names = np.array(vectorizer.get_feature_names_out())
        except ValueError:
            # Vectorizer error (e.g. empty vocabulary after stopword removal)
            summary = {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived English stopword removal"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        if len(feature_names) == 0:
            summary = {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        X_dense = X.toarray()

        # Step 2: Apply chosen filtering_method
        method_lower = filtering_method.lower()
        if method_lower == "variance":
            variances = np.var(X_dense, axis=0)
            low_p = np.percentile(variances, 5)
            high_p = np.percentile(variances, 95)
            kept_mask = (variances > low_p) & (variances < high_p)
        elif method_lower == "term_frequency":
            counts = np.sum(X_dense, axis=0)
            low_p = np.percentile(counts, 5)
            high_p = np.percentile(counts, 95)
            kept_mask = (counts > low_p) & (counts < high_p)
        else:  # tfidf
            transformer = TfidfTransformer()
            tfidf_mat = transformer.fit_transform(X_dense).toarray()
            mean_tfidf = np.mean(tfidf_mat, axis=0)
            p20 = np.percentile(mean_tfidf, 20)
            kept_mask = mean_tfidf >= p20

        kept_indices = np.where(kept_mask)[0]

        # Step 3: Handle zero terms kept cleanly
        if len(kept_indices) == 0:
            summary = {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        filtered_features = feature_names[kept_indices].tolist()
        filtered_X = X_dense[:, kept_indices]

        # Build DataFrame for DF2DB conversion (binary presence > 0)
        binary_df = pd.DataFrame(filtered_X > 0, columns=filtered_features)

        # Remove documents with no active terms
        binary_df = binary_df[binary_df.any(axis=1)].reset_index(drop=True)

        if binary_df.empty:
            summary = {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no non-empty transactions survived"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Step 4: Convert to PAMI Transactional Database via temp files
        temp_dir = tempfile.mkdtemp()
        db_file = os.path.join(temp_dir, "transactional_db.txt")

        try:
            converter = db_converter.DF2DB(binary_df)
            converter.convert2TransactionalDatabase(db_file, ">=", 1)

            alg_lower = algorithm.lower()
            patterns_list = []

            if alg_lower == "topk":
                topk_k = k if k is not None else 10
                obj = topk.FAE(iFile=db_file, k=topk_k, sep="\t")
                obj.startMine()
                pami_df = obj.getPatternsAsDataFrame()
            elif alg_lower == "maxfpgrowth":
                sup = min_sup if min_sup is not None else 1
                obj = maxfpgrowth.MaxFPGrowth(iFile=db_file, minSup=sup, sep="\t")
                obj.startMine()
                pami_df = obj.getPatternsAsDataFrame()
            else:  # fpgrowth
                sup = min_sup if min_sup is not None else 1
                obj = fpgrowth.FPGrowth(iFile=db_file, minSup=sup, sep="\t")
                obj.startMine()
                pami_df = obj.getPatternsAsDataFrame()

            if pami_df is not None and not pami_df.empty:
                # PAMI DataFrames have columns like 'Patterns' and 'Support'
                for _, row in pami_df.iterrows():
                    pat_str = str(row.iloc[0]).strip()
                    pat_items = [item.strip() for item in pat_str.split("\t") if item.strip()]
                    sup_val = int(row.iloc[1])
                    patterns_list.append({
                        "pattern": pat_items,
                        "support": sup_val
                    })
        finally:
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except Exception:
                    pass

        # Step 5: Store artifacts and result
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "kept_terms_count": len(filtered_features),
            "kept_terms": filtered_features,
            "patterns": patterns_list
        }

        session.store_result("mine_patterns_tool", {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k
        }, summary)

        return summary

    return [mine_patterns_tool]
