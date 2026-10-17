import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools

def test_dtm_tools_known_answers():
    # Single session reused for load_dataset_tool and tools_dtm
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load student DTM tools
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # Load test fixture
    dataset_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert dataset_res["n_documents"] == 8

    # Test build_dtm_tool (Known Answer #4)
    dtm_res = build_dtm_tool.invoke({})
    assert dtm_res["n_documents"] == 8
    assert dtm_res["n_terms"] == 5
    assert dtm_res["non_zero"] == 21
    assert dtm_res["total_elements"] == 40
    assert pytest.approx(dtm_res["sparsity_pct"], abs=1e-3) == 47.5
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]

    # Test term_frequency_tool (Known Answer #14)
    tf_res = term_frequency_tool.invoke({})
    expected_tf = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }
    assert tf_res["term_frequencies"] == expected_tf

    # Test dtm_heatmap_tool default params (Known Answer #21 - full matrix)
    heatmap_res_full = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})
    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert heatmap_res_full["n_documents"] == 8
    assert heatmap_res_full["n_terms"] == 5
    assert heatmap_res_full["feature_names"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert heatmap_res_full["matrix_slice"] == expected_full_matrix
    assert session.pending_figure is not None

    # Test dtm_heatmap_tool sliced (Known Answer #21 - top-left 3x2 slice)
    session.pending_figure = None
    heatmap_res_sliced = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    expected_sliced_matrix = [
        [3, 1],
        [2, 1],
        [1, 1]
    ]
    assert heatmap_res_sliced["n_documents"] == 3
    assert heatmap_res_sliced["n_terms"] == 2
    assert heatmap_res_sliced["feature_names"] == ["alpha", "always"]
    assert heatmap_res_sliced["matrix_slice"] == expected_sliced_matrix
    assert session.pending_figure is not None
