"""
Unit tests for TRLINKOS LLM Layer interface functions.

Tests cover:
- encode_text: text to embedding conversion
- reason_over_candidates: candidate reranking
- multi_step_reasoning: multi-turn reasoning with history
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trlinkos_llm_layer import (
    encode_text,
    reason_over_candidates,
    multi_step_reasoning,
    ReasoningConfig,
    TRLinkOSReasoningLayer,
    MockLLMAdapter,
    ChainOfThoughtAugmenter,
    create_reasoning_layer_for_llm,
)


class TestEncodeText:
    """Tests for encode_text function."""

    def test_basic_encoding(self):
        """Should encode text to correct dimension."""
        text = "Hello, world!"
        embedding = encode_text(text, embedding_dim=768)
        assert embedding.shape == (768,)

    def test_determinism(self):
        """Same text should produce same embedding."""
        text = "Test string"
        emb1 = encode_text(text, embedding_dim=512)
        emb2 = encode_text(text, embedding_dim=512)
        np.testing.assert_array_equal(emb1, emb2)

    def test_different_texts(self):
        """Different texts should produce different embeddings."""
        emb1 = encode_text("First text", embedding_dim=256)
        emb2 = encode_text("Second text", embedding_dim=256)
        assert not np.allclose(emb1, emb2)

    def test_normalization(self):
        """Embedding should be L2-normalized."""
        embedding = encode_text("Test", embedding_dim=768)
        norm = np.linalg.norm(embedding)
        np.testing.assert_almost_equal(norm, 1.0, decimal=4)

    def test_various_dimensions(self):
        """Should work with various embedding dimensions."""
        for dim in [64, 128, 256, 512, 768, 1024]:
            embedding = encode_text("Test text", embedding_dim=dim)
            assert embedding.shape == (dim,)

    def test_empty_text(self):
        """Should handle empty text."""
        embedding = encode_text("", embedding_dim=256)
        assert embedding.shape == (256,)

    def test_unicode_text(self):
        """Should handle unicode characters."""
        embedding = encode_text("日本語テスト 🎉", embedding_dim=256)
        assert embedding.shape == (256,)
        assert np.all(np.isfinite(embedding))

    def test_long_text(self):
        """Should handle long text."""
        long_text = "word " * 10000
        embedding = encode_text(long_text, embedding_dim=768)
        assert embedding.shape == (768,)
        assert np.all(np.isfinite(embedding))


class TestReasonOverCandidates:
    """Tests for reason_over_candidates function."""

    def test_basic_ranking(self):
        """Should return scores and best index."""
        query = np.random.randn(256)
        candidates = np.random.randn(5, 256)

        scores, best_idx = reason_over_candidates(
            query, candidates, reasoning_steps=2
        )

        assert scores.shape == (5,)
        assert 0 <= best_idx < 5

    def test_score_ordering(self):
        """Best index should have highest score."""
        query = np.random.randn(128)
        candidates = np.random.randn(4, 128)

        scores, best_idx = reason_over_candidates(
            query, candidates, reasoning_steps=2
        )

        assert scores[best_idx] == np.max(scores)

    def test_1d_query(self):
        """Should handle 1D query input."""
        query = np.random.randn(256)  # 1D
        candidates = np.random.randn(3, 256)

        scores, best_idx = reason_over_candidates(
            query, candidates, reasoning_steps=2
        )

        assert scores.shape == (3,)

    def test_2d_query(self):
        """Should handle 2D query input."""
        query = np.random.randn(1, 256)  # 2D
        candidates = np.random.randn(3, 256)

        scores, best_idx = reason_over_candidates(
            query, candidates, reasoning_steps=2
        )

        assert scores.shape == (3,)

    def test_single_candidate(self):
        """Should handle single candidate."""
        query = np.random.randn(128)
        candidates = np.random.randn(1, 128)

        scores, best_idx = reason_over_candidates(
            query, candidates, reasoning_steps=2
        )

        assert scores.shape == (1,)
        assert best_idx == 0

    def test_custom_reasoning_layer(self):
        """Should accept custom reasoning layer."""
        config = ReasoningConfig(
            input_dim=128,
            output_dim=64,
            z_dim=32,
            hidden_dim=64,
            num_experts=2,
            max_reasoning_steps=3,
            project_to_llm_dim=True,
        )
        layer = TRLinkOSReasoningLayer(config)

        query = np.random.randn(128)
        candidates = np.random.randn(4, 128)

        scores, best_idx = reason_over_candidates(
            query, candidates, reasoning_layer=layer, reasoning_steps=3
        )

        assert scores.shape == (4,)


class TestMultiStepReasoning:
    """Tests for multi_step_reasoning function."""

    def test_empty_history(self):
        """Should work with empty history (first turn)."""
        new_input = np.random.randn(256)
        output, meta = multi_step_reasoning([], new_input)

        assert output.shape == (256,)
        assert meta["num_history_items"] == 0
        assert "total_reasoning_steps" in meta
        assert "dag_nodes" in meta

    def test_with_history(self):
        """Should work with history."""
        history = [np.random.randn(256) for _ in range(3)]
        new_input = np.random.randn(256)

        output, meta = multi_step_reasoning(history, new_input)

        assert output.shape == (256,)
        assert meta["num_history_items"] == 3

    def test_context_window_limiting(self):
        """Should respect context_window limit."""
        history = [np.random.randn(128) for _ in range(10)]
        new_input = np.random.randn(128)

        output, meta = multi_step_reasoning(
            history, new_input, context_window=5
        )

        assert meta["num_history_items"] == 5

    def test_2d_input(self):
        """Should handle 2D new_input."""
        new_input = np.random.randn(1, 256)  # 2D
        output, meta = multi_step_reasoning([], new_input)

        assert output.shape == (256,)

    def test_metadata_contents(self):
        """Should return expected metadata keys."""
        new_input = np.random.randn(128)
        output, meta = multi_step_reasoning([], new_input)

        expected_keys = [
            "num_history_items",
            "total_reasoning_steps",
            "best_score",
            "dag_nodes",
            "depth_stats",
        ]
        for key in expected_keys:
            assert key in meta, f"Missing key: {key}"

    def test_custom_reasoning_layer(self):
        """Should accept custom reasoning layer."""
        config = ReasoningConfig(
            input_dim=128,
            output_dim=64,
            z_dim=32,
            hidden_dim=64,
            num_experts=2,
            max_reasoning_steps=4,
            project_to_llm_dim=True,
        )
        layer = TRLinkOSReasoningLayer(config)

        new_input = np.random.randn(128)
        output, meta = multi_step_reasoning([], new_input, reasoning_layer=layer)

        assert output.shape == (128,)

    def test_incremental_history(self):
        """Should work with incrementally built history."""
        history = []
        outputs = []

        for i in range(3):
            new_input = np.random.randn(128)
            output, meta = multi_step_reasoning(history, new_input)
            history.append(output)
            outputs.append(output)

            assert output.shape == (128,)
            assert meta["num_history_items"] == i


class TestIntegration:
    """Integration tests for LLM layer components."""

    def test_encode_and_reason(self):
        """Should work end-to-end with encoded text."""
        # Encode texts
        query_emb = encode_text("What is machine learning?", embedding_dim=256)
        candidates = [
            encode_text("ML is a type of AI.", embedding_dim=256),
            encode_text("The sky is blue.", embedding_dim=256),
            encode_text("ML enables computers to learn from data.", embedding_dim=256),
        ]
        candidate_embs = np.array(candidates)

        # Reason over candidates
        scores, best_idx = reason_over_candidates(
            query_emb, candidate_embs, reasoning_steps=3
        )

        assert scores.shape == (3,)
        assert 0 <= best_idx < 3

    def test_multi_turn_conversation(self):
        """Should handle multi-turn conversation."""
        queries = [
            "What is AI?",
            "How does it work?",
            "Give me an example.",
        ]

        history = []
        for query in queries:
            query_emb = encode_text(query, embedding_dim=256)
            output, meta = multi_step_reasoning(history, query_emb)
            history.append(output)

        assert len(history) == 3
        for h in history:
            assert h.shape == (256,)

    def test_create_reasoning_layer_factory(self):
        """Factory function should create valid layers."""
        layer, config = create_reasoning_layer_for_llm("gpt2")
        assert config.input_dim == 768

        layer, config = create_reasoning_layer_for_llm("llama-7b")
        assert config.input_dim == 4096

    def test_chain_of_thought_augmenter(self):
        """ChainOfThoughtAugmenter should work with new functions."""
        config = ReasoningConfig(input_dim=256, output_dim=128, max_reasoning_steps=3)
        reasoning_layer = TRLinkOSReasoningLayer(config)
        cot = ChainOfThoughtAugmenter(reasoning_layer)

        # Add thoughts
        thought1 = encode_text("First thought", embedding_dim=256)
        thought2 = encode_text("Second thought", embedding_dim=256)

        enhanced1, trace1 = cot.add_thought(thought1, "First thought")
        enhanced2, trace2 = cot.add_thought(thought2, "Second thought")

        chain_trace = cot.get_chain_trace()
        assert len(chain_trace) == 2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_encode_special_characters(self):
        """Should handle special characters."""
        texts = [
            "Special chars: !@#$%^&*()",
            "Newlines:\n\ntest",
            "Tabs:\t\ttabs",
            "Mixed: 你好 Hello مرحبا",
        ]
        for text in texts:
            emb = encode_text(text, embedding_dim=128)
            assert emb.shape == (128,)
            assert np.all(np.isfinite(emb))

    def test_reason_with_identical_candidates(self):
        """Should handle identical candidates."""
        query = np.random.randn(128)
        candidate = np.random.randn(128)
        candidates = np.tile(candidate, (3, 1))

        scores, best_idx = reason_over_candidates(
            query, candidates, reasoning_steps=2
        )

        # All scores should be similar since candidates are identical
        assert scores.shape == (3,)

    def test_numerical_stability(self):
        """Should handle extreme values."""
        # Test with large values
        query = np.ones(128) * 1000
        candidates = np.ones((3, 128)) * 1000

        scores, best_idx = reason_over_candidates(
            query, candidates, reasoning_steps=2
        )
        assert np.all(np.isfinite(scores))

        # Test with small values
        query = np.ones(128) * 1e-10
        candidates = np.ones((3, 128)) * 1e-10

        scores, best_idx = reason_over_candidates(
            query, candidates, reasoning_steps=2
        )
        assert np.all(np.isfinite(scores))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
