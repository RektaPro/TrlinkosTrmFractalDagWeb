"""
Tests for new implementations from ANALYSE_COMPARATIVE_HONNETE_LLM.md

This module tests:
- TorqueRouter.forward_sparse (sparse top-k routing)
- DivergenceDetector (reasoning divergence detection)
- DAGVisualizer (DAG visualization)
- BenchmarkSuite (formal benchmarks)
- AdvancedLLMIntegration (advanced LLM integration)
"""

import sys
import os
import tempfile
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_sparse_routing():
    """Test TorqueRouter.forward_sparse method."""
    from t_rlinkos_trm_fractal_dag import TorqueRouter

    np.random.seed(42)
    router = TorqueRouter(x_dim=32, y_dim=16, z_dim=32, num_experts=4)

    x = np.random.randn(8, 32)
    y = np.random.randn(8, 16)
    z = np.random.randn(8, 32)

    # Test basic functionality
    sparse_weights, top_indices = router.forward_sparse(x, y, z, top_k=2)

    # Check shapes
    assert sparse_weights.shape == (8, 4), f"Wrong weights shape: {sparse_weights.shape}"
    assert top_indices.shape == (8, 2), f"Wrong indices shape: {top_indices.shape}"

    # Check normalization
    weight_sums = sparse_weights.sum(axis=-1)
    assert np.allclose(weight_sums, 1.0), f"Weights not normalized: {weight_sums}"

    # Check sparsity
    active_counts = np.sum(sparse_weights > 0, axis=-1)
    assert np.all(active_counts == 2), f"Wrong number of active experts: {active_counts}"

    # Check determinism
    sparse_weights2, top_indices2 = router.forward_sparse(x, y, z, top_k=2)
    assert np.allclose(sparse_weights, sparse_weights2), "Sparse routing not deterministic"

    print("✅ test_sparse_routing passed!")


def test_divergence_detector():
    """Test DivergenceDetector class."""
    from t_rlinkos_trm_fractal_dag import DivergenceDetector

    # Test 1: No divergence on stable sequence
    detector1 = DivergenceDetector(variance_threshold=0.1)
    for i in range(5):
        detector1.update(score=0.5 + i * 0.01, state=np.ones((1, 16)) * (0.5 + i * 0.01))
    is_div, reason = detector1.is_diverging()
    assert not is_div, f"False positive on stable sequence: {reason}"

    # Test 2: Detect high variance
    detector2 = DivergenceDetector(variance_threshold=0.05)
    for s in [0.1, 0.9, 0.2, 0.8, 0.3]:
        detector2.update(score=s, state=np.ones((1, 16)) * s)
    is_div2, reason2 = detector2.is_diverging()
    assert is_div2, "Failed to detect high variance"
    assert "variance" in reason2.lower(), f"Wrong reason: {reason2}"

    # Test 3: Detect negative gradient
    detector3 = DivergenceDetector(gradient_threshold=-0.05)
    for i in range(5):
        detector3.update(score=1.0 - i * 0.1, state=np.ones((1, 16)))
    is_div3, reason3 = detector3.is_diverging()
    assert is_div3, "Failed to detect negative gradient"
    assert "gradient" in reason3.lower() or "degradation" in reason3.lower(), f"Wrong reason: {reason3}"

    # Test 4: Reset works
    detector4 = DivergenceDetector()
    detector4.update(0.5, np.ones((1, 16)))
    detector4.reset()
    stats = detector4.get_statistics()
    assert stats["num_observations"] == 0, "Reset didn't clear history"

    # Test 5: Get statistics
    detector5 = DivergenceDetector()
    detector5.update(0.4, np.ones((1, 16)))
    detector5.update(0.5, np.ones((1, 16)))
    detector5.update(0.6, np.ones((1, 16)))
    stats5 = detector5.get_statistics()
    assert stats5["num_observations"] == 3
    assert abs(stats5["score_mean"] - 0.5) < 0.01

    print("✅ test_divergence_detector passed!")


def test_dag_visualizer():
    """Test DAGVisualizer class."""
    from dag_visualizer import DAGVisualizer
    from t_rlinkos_trm_fractal_dag import TRLinkosTRM, FractalMerkleDAG

    np.random.seed(42)

    # Create a model and run reasoning
    model = TRLinkosTRM(x_dim=32, y_dim=16, z_dim=32, hidden_dim=64, num_experts=2)
    x_batch = np.random.randn(2, 32)
    target = np.random.randn(2, 16)

    def scorer(x, y):
        return -np.mean((y - target) ** 2, axis=-1)

    y_pred, dag = model.forward_recursive(x_batch, max_steps=5, scorer=scorer)

    # Create visualizer
    viz = DAGVisualizer(dag)

    # Use tempfile for cross-platform compatibility
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test to_html
        html_path = os.path.join(tmpdir, "test_viz_dag.html")
        result = viz.to_html(html_path)
        assert os.path.exists(html_path), "HTML file not created"

        with open(html_path, 'r') as f:
            content = f.read()
        assert "T-RLINKOS" in content, "HTML missing title"
        assert "d3" in content.lower(), "HTML missing D3.js"

        # Test to_graphml
        graphml_path = os.path.join(tmpdir, "test_viz_dag.graphml")
        viz.to_graphml(graphml_path)
        assert os.path.exists(graphml_path), "GraphML file not created"

        # Test to_dot
        dot_path = os.path.join(tmpdir, "test_viz_dag.dot")
        viz.to_dot(dot_path)
        assert os.path.exists(dot_path), "DOT file not created"

    # Test to_json (no file output, just dict)
    json_data = viz.to_json()
    assert "metadata" in json_data
    assert "nodes" in json_data
    assert "edges" in json_data
    assert json_data["metadata"]["total_nodes"] == len(dag.nodes)

    # Test explain_path
    explanation = viz.explain_path()
    assert "T-RLINKOS" in explanation
    assert "STEP" in explanation

    # Test get_summary
    summary = viz.get_summary()
    assert "total_nodes" in summary
    assert "best_score" in summary
    assert summary["total_nodes"] == len(dag.nodes)

    # Test with empty DAG
    empty_dag = FractalMerkleDAG()
    empty_viz = DAGVisualizer(empty_dag)
    empty_explanation = empty_viz.explain_path()
    assert "No reasoning path available" in empty_explanation

    print("✅ test_dag_visualizer passed!")


def test_benchmark_suite():
    """Test BenchmarkSuite class."""
    from benchmarks.formal_benchmarks import BenchmarkSuite, BenchmarkResult

    # Test individual benchmarks
    result = BenchmarkSuite.benchmark_sparse_routing()
    assert isinstance(result, BenchmarkResult)
    assert result.benchmark == "Sparse Routing"
    assert result.status in ["PASS", "FAIL", "NEUTRAL"]
    assert 0.0 <= result.score <= 1.0

    result2 = BenchmarkSuite.benchmark_divergence_detection()
    assert result2.benchmark == "Divergence Detection"

    result3 = BenchmarkSuite.benchmark_auditability()
    assert result3.benchmark == "Cryptographic Auditability"

    # Test results_to_dict
    results = [result, result2, result3]
    results_dict = BenchmarkSuite.results_to_dict(results)
    assert "summary" in results_dict
    assert "benchmarks" in results_dict
    assert results_dict["summary"]["total_benchmarks"] == 3

    print("✅ test_benchmark_suite passed!")


def test_advanced_llm_integration():
    """Test AdvancedLLMIntegration class."""
    from trlinkos_llm_layer import (
        AdvancedLLMIntegration,
        TRLinkOSReasoningLayer,
        ReasoningConfig,
        MockLLMAdapter,
    )

    np.random.seed(42)

    # Create components
    config = ReasoningConfig(
        input_dim=768,
        output_dim=256,
        z_dim=128,
        max_reasoning_steps=4,
        project_to_llm_dim=False,
    )
    reasoning = TRLinkOSReasoningLayer(config)
    adapter = MockLLMAdapter(hidden_dim=768)
    integration = AdvancedLLMIntegration(reasoning, adapter)

    # Test reason_and_generate
    text, dag, meta = integration.reason_and_generate("Test prompt for reasoning")
    assert isinstance(text, str)
    assert len(text) > 0
    assert "reasoning_steps" in meta
    assert "best_score" in meta
    assert meta["verified"] is True

    # Test chain_of_thought
    chain_results = integration.chain_of_thought("Test problem", num_steps=3)
    assert len(chain_results) == 3
    assert all("step" in r for r in chain_results)
    assert all("dag_nodes" in r for r in chain_results)

    # Test get_chain_summary
    summary = integration.get_chain_summary()
    assert summary["total_steps"] == 3
    assert "divergence_count" in summary
    assert "mean_score" in summary

    # Test reset
    integration.reset()
    summary_after_reset = integration.get_chain_summary()
    assert summary_after_reset["total_steps"] == 0

    # Test without adapter (should raise error)
    integration_no_adapter = AdvancedLLMIntegration(reasoning, None)
    try:
        integration_no_adapter.reason_and_generate("test")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected

    print("✅ test_advanced_llm_integration passed!")


def run_all_new_tests():
    """Run all tests for new implementations."""
    print("=" * 60)
    print("Testing New Implementations from ANALYSE_COMPARATIVE_HONNETE_LLM.md")
    print("=" * 60)

    tests = [
        test_sparse_routing,
        test_divergence_detector,
        test_dag_visualizer,
        test_benchmark_suite,
        test_advanced_llm_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} FAILED: {e}")

    print()
    print("=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("🎉 ALL NEW IMPLEMENTATION TESTS PASSED! 🎉")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_new_tests()
    sys.exit(0 if success else 1)
