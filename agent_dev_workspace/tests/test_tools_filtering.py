import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools

def test_variance_filter_tool_known_answer_5():
    # Instantiate session state once
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build initial DTM (8 documents, 5 terms: alpha, always, beta, delta, gamma)
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"vectorizer_type": "count"})

    # Get variance_filter_tool bound to same session
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Known Answer #5:
    # Threshold = 0.15 keeps alpha (1.1875), beta (0.234375), delta (0.25), gamma (1.109375)
    # and removes always (0.109375)
    result = variance_filter_tool.invoke({"threshold": 0.15})

    assert result["initial_features"] == 5
    assert result["kept_features"] == 4
    assert result["removed_features"] == 1
    assert result["removed_feature_names"] == ["always"]
    assert result["kept_feature_names"] == ["alpha", "beta", "delta", "gamma"]

    # Check updated session state
    assert session.feature_names == ["alpha", "beta", "delta", "gamma"]
    assert session.feature_matrix.shape == (8, 4)

    # Verify full_report DataFrame in results_store
    result_id = result["result_id"]
    stored_data = session.results_store[result_id]
    report_df = stored_data["full_report"]

    assert report_df.columns.tolist() == ["term", "variance"]
    # Check that alpha has highest variance (~1.1875) and always has lowest (~0.109375)
    top_row = report_df.iloc[0]
    bottom_row = report_df.iloc[-1]
    assert top_row["term"] == "alpha"
    assert pytest.approx(top_row["variance"], abs=1e-3) == 1.1875
    assert bottom_row["term"] == "always"
    assert pytest.approx(bottom_row["variance"], abs=1e-3) == 0.109375

def test_pearson_filter_tool_known_answer_6():
    # Instantiate session state fresh per test function
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build initial DTM (8 documents, 5 terms)
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"vectorizer_type": "count"})

    # Get pearson_filter_tool bound to same session
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Known Answer #6: target_class="catB"
    # alpha: -0.6882
    # always: 0.3780
    # beta: -0.7746
    # delta: 1.0000
    # gamma: 0.8307
    result = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    corrs = result["correlations"]
    assert pytest.approx(corrs["alpha"], abs=1e-3) == -0.6882
    assert pytest.approx(corrs["always"], abs=1e-3) == 0.3780
    assert pytest.approx(corrs["beta"], abs=1e-3) == -0.7746
    assert pytest.approx(corrs["delta"], abs=1e-3) == 1.0000
    assert pytest.approx(corrs["gamma"], abs=1e-3) == 0.8307

    # Verify full_report DataFrame format
    result_id = result["result_id"]
    report_df = session.results_store[result_id]["full_report"]
    assert report_df.columns.tolist() == ["term", "pearson_r"]
    # Sorted descending: top term should be delta (1.0000)
    assert report_df.iloc[0]["term"] == "delta"
    assert pytest.approx(report_df.iloc[0]["pearson_r"], abs=1e-3) == 1.0000

def test_spearman_filter_tool_known_answer_6():
    # Instantiate session state fresh per test function
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build initial DTM (8 documents, 5 terms)
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"vectorizer_type": "count"})

    # Get spearman_filter_tool bound to same session
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Known Answer #6: target_class="catB"
    # alpha: -0.7500
    # always: 0.3780
    # beta: -0.7746
    # delta: 1.0000
    # gamma: 0.9363
    result = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    corrs = result["correlations"]
    assert pytest.approx(corrs["alpha"], abs=1e-3) == -0.7500
    assert pytest.approx(corrs["always"], abs=1e-3) == 0.3780
    assert pytest.approx(corrs["beta"], abs=1e-3) == -0.7746
    assert pytest.approx(corrs["delta"], abs=1e-3) == 1.0000
    assert pytest.approx(corrs["gamma"], abs=1e-3) == 0.9363

    # Verify full_report DataFrame format
    result_id = result["result_id"]
    report_df = session.results_store[result_id]["full_report"]
    assert report_df.columns.tolist() == ["term", "spearman_r"]
    # Sorted descending: top term should be delta (1.0000)
    assert report_df.iloc[0]["term"] == "delta"
    assert pytest.approx(report_df.iloc[0]["spearman_r"], abs=1e-3) == 1.0000
