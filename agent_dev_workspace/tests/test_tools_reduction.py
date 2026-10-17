import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools

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

def test_reduce_dimensions_pca():
    session = setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Known Answer #7: PCA reduction (n_components=2)
    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    assert "coordinates" in res
    assert "explained_variance_ratio" in res
    coords = np.array(res["coordinates"])
    
    # Expected PCA coordinates from TEST_FIXTURE.md Known Answer #7
    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])
    expected_var = [0.7532, 0.1965]

    assert np.allclose(coords, expected_coords, atol=1e-3)
    assert np.allclose(res["explained_variance_ratio"], expected_var, atol=1e-3)
    assert session.pending_figure is not None

def test_reduce_dimensions_tsne_properties():
    session = setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # t-SNE reduction call
    res = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2.0,
        "random_state": 42
    })

    assert "coordinates" in res
    coords = np.array(res["coordinates"])
    assert coords.shape == (8, 2)

    # Verify duplicate documents (rows 5 and 6) have identical coordinates
    assert np.allclose(coords[5], coords[6], atol=1e-3)
    assert session.pending_figure is not None

def test_binarize_labels_tool():
    session = setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_labels_tool.invoke({})
    assert "binarized_columns" in res
    assert res["n_rows"] == 8
    
    df = session.dataframe
    assert "cat_catA" in df.columns or "cat_catB" in df.columns
