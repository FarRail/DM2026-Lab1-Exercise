import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_tokenize_tool_known_answer_10():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load test fixture
    dataset_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert dataset_res["n_documents"] == 8

    # Load student data tools
    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    # Invoke tokenize_tool
    result = tokenize_tool.invoke({})
    assert result["n_documents"] == 8
    assert result["total_tokens"] == 27
    assert pytest.approx(result["avg_tokens_per_doc"], abs=1e-3) == 3.375

    # Check session state dataframe unigrams column against Known Answer #10
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]

    unigrams = session.dataframe["unigrams"].tolist()
    assert unigrams == expected_tokens
    assert session.artifacts["unigrams"] == expected_tokens

def test_list_files_tool_known_answer_11():
    session = SessionState()

    # Load student data tools
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # Known Answer #11: subdirectory="newdataset" returns directories: [], files: ["Reddit-stock-sentiment.csv"]
    result = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert result["subdirectory"] == "newdataset"
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]

def test_inspect_data_tool_custom_design():
    session = SessionState()

    # Load premade load_dataset_tool and sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Load student inspect_data_tool
    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # Test head mode with n_rows=3
    head_res = inspect_data_tool.invoke({"n_rows": 3, "mode": "head"})
    assert head_res["total_rows"] == 8
    assert head_res["inspected_rows"] == 3
    assert head_res["mode"] == "head"
    assert "text" in head_res["columns"]
    assert head_res["data"][0]["text"] == "always alpha alpha alpha beta"

    # Test tail mode with n_rows=2
    tail_res = inspect_data_tool.invoke({"n_rows": 2, "mode": "tail"})
    assert tail_res["total_rows"] == 8
    assert tail_res["inspected_rows"] == 2
    assert tail_res["mode"] == "tail"
    # Row 7 in fixture is empty text
    assert tail_res["data"][-1]["text"] == ""

def test_check_missing_tool_known_answer_2():
    session = SessionState()

    # Load premade load_dataset_tool and sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Load student check_missing_tool
    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    # Known Answer #2: exactly 1 row has missing/empty text (row 7)
    result = check_missing_tool.invoke({})
    assert result["total_rows"] == 8
    assert result["missing_count"] == 1
    assert result["missing_indices"] == [7]

def test_check_duplicates_tool_known_answer_3():
    session = SessionState()

    # Load premade load_dataset_tool and sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Load student check_duplicates_tool
    data_tools = make_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    # Known Answer #3 check without drop: exactly 1 duplicate (row 6 is copy of row 5)
    res_check = check_duplicates_tool.invoke({"drop": False})
    assert res_check["total_rows_before"] == 8
    assert res_check["duplicate_count"] == 1
    assert res_check["duplicate_indices"] == [6]
    assert res_check["total_rows_after"] == 8

    # Known Answer #3 with drop=True: keep=False drops every copy of duplicated row, leaving 6 rows
    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert res_drop["total_rows_before"] == 8
    assert res_drop["duplicate_count"] == 1
    assert res_drop["total_rows_after"] == 6
    assert len(session.dataframe) == 6

def test_sample_data_tool_known_answer_13():
    session = SessionState()

    # Load premade load_dataset_tool and sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Load student sample_data_tool
    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    # Known Answer #13: n=4, random_state=42 returns exactly sampled_indices [1, 5, 0, 7]
    result = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert result["total_rows_before"] == 8
    assert result["sampled_rows"] == 4
    assert result["sampled_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == 4

def test_describe_data_tool_known_answer_18():
    session = SessionState()

    # Load premade load_dataset_tool and sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Load student describe_data_tool
    data_tools = make_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    result = describe_data_tool.invoke({})
    assert result["n_documents"] == 8

    # Overall stats assertions against Known Answer #18
    # count=8, mean=19.875, std=9.4330, min=0, 25%=17.75, 50%=20.5, 75%=25.25, max=30
    overall = result["overall_stats"]
    assert overall["count"] == 8
    assert pytest.approx(overall["mean"], abs=1e-3) == 19.875
    assert pytest.approx(overall["std"], abs=1e-3) == 9.4330
    assert overall["min"] == 0
    assert pytest.approx(overall["25%"], abs=1e-3) == 17.75
    assert pytest.approx(overall["50%"], abs=1e-3) == 20.5
    assert pytest.approx(overall["75%"], abs=1e-3) == 25.25
    assert overall["max"] == 30

    # Per-category stats assertions against Known Answer #18
    # catA: count=4, mean=17.25, std=12.5, min=0, 25%=12.75, 50%=20.0, 75%=24.5, max=29
    catA = result["per_category_stats"]["catA"]
    assert catA["count"] == 4
    assert pytest.approx(catA["mean"], abs=1e-3) == 17.25
    assert pytest.approx(catA["std"], abs=1e-3) == 12.5
    assert catA["min"] == 0
    assert pytest.approx(catA["25%"], abs=1e-3) == 12.75
    assert pytest.approx(catA["50%"], abs=1e-3) == 20.0
    assert pytest.approx(catA["75%"], abs=1e-3) == 24.5
    assert catA["max"] == 29

    # catB: count=4, mean=22.5, std=5.7446, min=18, 25%=18.0, 50%=21.0, 75%=25.5, max=30
    catB = result["per_category_stats"]["catB"]
    assert catB["count"] == 4
    assert pytest.approx(catB["mean"], abs=1e-3) == 22.5
    assert pytest.approx(catB["std"], abs=1e-3) == 5.7446
    assert catB["min"] == 18
    assert pytest.approx(catB["25%"], abs=1e-3) == 18.0
    assert pytest.approx(catB["50%"], abs=1e-3) == 21.0
    assert pytest.approx(catB["75%"], abs=1e-3) == 25.5
    assert catB["max"] == 30

    # Confirm box plot figure assigned to session.pending_figure
    assert session.pending_figure is not None
