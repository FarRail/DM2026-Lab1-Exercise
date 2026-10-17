import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools

def test_mine_patterns_tool_tfidf_fpgrowth():
    session = SessionState()
    
    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_result["n_documents"] == 8

    # Import student's mine_patterns_tool directly
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Test Known Answer #17: TF-IDF filtering with FPGrowth algorithm per category
    # 1. Test catA -> kept term alpha, pattern {alpha} with support 3
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "min_sup": 1,
        "algorithm": "fpgrowth"
    })

    assert "patterns" in res_catA
    cata_patterns = res_catA["patterns"]
    alpha_pattern = next((p for p in cata_patterns if set(p.get("pattern", [])) == {"alpha"}), None)
    assert alpha_pattern is not None, f"Expected {{'alpha'}} pattern in catA, got: {cata_patterns}"
    assert alpha_pattern["support"] == 3

    # 2. Test catB -> kept term gamma, pattern {gamma} with support 4
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "min_sup": 1,
        "algorithm": "fpgrowth"
    })

    assert "patterns" in res_catB
    catb_patterns = res_catB["patterns"]
    gamma_pattern = next((p for p in catb_patterns if set(p.get("pattern", [])) == {"gamma"}), None)
    assert gamma_pattern is not None, f"Expected {{'gamma'}} pattern in catB, got: {catb_patterns}"
    assert gamma_pattern["support"] == 4

def test_mine_patterns_tool_maxfpgrowth():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Test Known Answer #20: MaxFPGrowth algorithm returns zero patterns on this single term DB
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "min_sup": 1,
        "algorithm": "maxfpgrowth"
    })

    assert "patterns" in res_catA
    assert len(res_catA["patterns"]) == 0

    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "min_sup": 1,
        "algorithm": "maxfpgrowth"
    })

    assert "patterns" in res_catB
    assert len(res_catB["patterns"]) == 0
