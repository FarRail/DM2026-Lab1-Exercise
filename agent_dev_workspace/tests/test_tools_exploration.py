import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools

def setup_session_with_dtm():
    session = SessionState()
    
    # 1. Load sample dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # 2. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    return session

def test_cosine_similarity_tool_known_answer_16():
    session = setup_session_with_dtm()
    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Known Answer #16 pairs from TEST_FIXTURE.md
    # Doc 0 vs 1 (both catA, near-identical text): 0.9847
    res_0_1 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_0_1["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Doc 0 vs 3 (catA vs catB, share only "always"): 0.0909
    res_0_3 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert res_0_3["cosine_similarity"] == pytest.approx(0.0909, abs=1e-3)

    # Doc 3 vs 4 (both catB, near-identical text): 0.9847
    res_3_4 = cosine_sim_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert res_3_4["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Doc 0 vs 7 (catA vs empty text row): 0.0
    res_0_7 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert res_0_7["cosine_similarity"] == 0.0

def test_feature_correlation_matrix_tool_known_answer_19():
    session = setup_session_with_dtm()
    exploration_tools = make_exploration_tools(session)
    feat_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    # Known Answer #19: Feature correlation matrix across top variance terms
    res = feat_corr_tool.invoke({})

    assert "terms" in res
    assert "correlation_matrix" in res
    assert session.pending_figure is not None

    terms = res["terms"]
    corr_matrix = np.array(res["correlation_matrix"])

    # Expected term order by variance descending: alpha, gamma, delta, beta, always
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert terms == expected_terms

    # Check diagonal is all 1.0
    assert np.allclose(np.diag(corr_matrix), 1.0)

    # Known correlation checks from TEST_FIXTURE.md Known Answer #19
    # alpha vs beta (indices 0 and 3): 0.8885
    assert corr_matrix[0, 3] == pytest.approx(0.8885, abs=1e-3)

    # gamma vs delta (indices 1 and 2): 0.8307
    assert corr_matrix[1, 2] == pytest.approx(0.8307, abs=1e-3)

    # alpha vs delta (indices 0 and 2): -0.6882
    assert corr_matrix[0, 2] == pytest.approx(-0.6882, abs=1e-3)
