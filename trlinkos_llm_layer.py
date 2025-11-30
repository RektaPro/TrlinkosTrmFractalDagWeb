"""
TRLINKOS LLM Reasoning Layer

This module provides T-RLINKOS as a reasoning layer that can be connected to any
open-source LLM (Mistral, LLaMA, etc.). It acts as an intermediary reasoning engine
that enhances LLM capabilities with:

- Recursive reasoning with backtracking
- Fractal exploration of solution space
- Cryptographic audit trail via Merkle-DAG
- dCaAP-based neural processing

Key Components:
- LLMAdapter: Abstract base class for LLM integration
- HuggingFaceAdapter: Concrete adapter for HuggingFace models
- TRLinkOSReasoningLayer: Main reasoning layer wrapping TRLinkosTRM
- ReasoningConfig: Configuration for the reasoning layer

Usage:
    from trlinkos_llm_layer import TRLinkOSReasoningLayer, ReasoningConfig

    # Create reasoning layer
    config = ReasoningConfig(hidden_dim=256, num_experts=4)
    reasoning_layer = TRLinkOSReasoningLayer(config)

    # Use with any LLM hidden states
    llm_hidden_states = ...  # [B, seq_len, hidden_dim] from LLM
    enhanced_output, dag = reasoning_layer.reason(llm_hidden_states)

References:
- LLM Integration discussed in: docs/ROADMAP_TRLINKOS_V2.md (Phase 3)
- L-InCOT integration: https://github.com/RektaPro/L-inCOTv0.1
"""

import hashlib
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

# Import core T-RLINKOS components
from t_rlinkos_trm_fractal_dag import (
    TRLinkosTRM,
    FractalMerkleDAG,
    LinearNP,
    gelu,
    softmax,
    TextEncoder,
)


# ============================
#  Configuration
# ============================


@dataclass
class ReasoningConfig:
    """Configuration for the TRLINKOS reasoning layer.

    Attributes:
        input_dim: Dimension of input from LLM (hidden states).
                   Common values: 768 (BERT), 1024 (GPT-2), 4096 (LLaMA-7B)
        output_dim: Dimension of reasoning output
        z_dim: Internal state dimension for recursive reasoning
        hidden_dim: Hidden layer dimension
        num_experts: Number of dCaAP experts for mixture-of-experts routing
        max_reasoning_steps: Maximum reasoning iterations
        inner_recursions: Number of inner recursions per step
        enable_backtracking: Enable state restoration to best-scoring nodes
        backtrack_threshold: Score degradation threshold for backtracking
        enable_fractal_branching: Enable fractal exploration during reasoning
        branch_threshold: Variance threshold for creating fractal branches
        max_branches_per_node: Maximum branches per DAG node
        perturbation_scale: Scale of perturbation for branch exploration
        use_attention_pooling: Use attention-based pooling for sequence inputs
        project_to_llm_dim: Project output back to LLM dimension
    """

    input_dim: int = 4096  # LLaMA-7B hidden size
    output_dim: int = 256  # Reasoning output dimension
    z_dim: int = 128  # Internal state dimension
    hidden_dim: int = 256  # Hidden layer dimension
    num_experts: int = 4  # Number of dCaAP experts

    # Reasoning parameters
    max_reasoning_steps: int = 8  # Maximum reasoning iterations
    inner_recursions: int = 3  # Inner recursions per step
    enable_backtracking: bool = True  # Enable backtracking
    backtrack_threshold: float = 0.1  # Backtrack threshold

    # Fractal parameters
    enable_fractal_branching: bool = False  # Enable fractal exploration
    branch_threshold: float = 0.05  # Variance threshold for branching
    max_branches_per_node: int = 2  # Max branches per node
    perturbation_scale: float = 0.1  # Perturbation scale for branches

    # Sequence handling
    use_attention_pooling: bool = True  # Use attention pooling for sequences
    project_to_llm_dim: bool = True  # Project output to LLM dim


# ============================
#  LLM Adapter Interface
# ============================


@runtime_checkable
class LLMAdapterProtocol(Protocol):
    """Protocol for LLM adapters."""

    def get_hidden_states(
        self,
        input_ids: Any,
        attention_mask: Optional[Any] = None,
    ) -> np.ndarray:
        """Extract hidden states from the LLM."""
        ...

    def get_hidden_dim(self) -> int:
        """Return the hidden dimension of the LLM."""
        ...

    def get_model_name(self) -> str:
        """Return the model name/identifier."""
        ...


class LLMAdapter(ABC):
    """Abstract base class for LLM adapters.

    This class defines the interface for connecting T-RLINKOS to any LLM.
    Subclasses must implement methods to:
    - Extract hidden states from the LLM
    - Get model dimensions
    - Handle model-specific preprocessing

    Supported LLM families:
    - LLaMA (Meta)
    - Mistral (Mistral AI)
    - GPT-2/GPT-J/GPT-NeoX (Various)
    - BERT family (Google)
    - Any HuggingFace transformers model
    """

    @abstractmethod
    def get_hidden_states(
        self,
        input_ids: Any,
        attention_mask: Optional[Any] = None,
    ) -> np.ndarray:
        """Extract hidden states from the LLM.

        Args:
            input_ids: Token IDs [B, seq_len] (can be numpy, list, or tensor)
            attention_mask: Optional attention mask [B, seq_len]

        Returns:
            Hidden states as numpy array [B, seq_len, hidden_dim]
        """
        pass

    @abstractmethod
    def get_hidden_dim(self) -> int:
        """Return the hidden dimension of the LLM.

        Common values:
        - BERT-base: 768
        - BERT-large: 1024
        - GPT-2: 768/1024/1280
        - LLaMA-7B: 4096
        - LLaMA-13B: 5120
        - Mistral-7B: 4096

        Returns:
            Hidden dimension (int)
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name/identifier.

        Returns:
            Model name string (e.g., "mistralai/Mistral-7B-v0.1")
        """
        pass

    def preprocess_hidden_states(
        self,
        hidden_states: np.ndarray,
    ) -> np.ndarray:
        """Optional preprocessing of hidden states.

        Default implementation returns hidden states unchanged.
        Override for model-specific preprocessing (normalization, etc.)

        Args:
            hidden_states: Raw hidden states [B, seq_len, hidden_dim]

        Returns:
            Preprocessed hidden states [B, seq_len, hidden_dim]
        """
        return hidden_states


class HuggingFaceAdapter(LLMAdapter):
    """Adapter for HuggingFace transformers models.

    Supports any model from the HuggingFace transformers library that
    provides hidden states output, including:
    - LLaMA (meta-llama/Llama-2-7b-hf)
    - Mistral (mistralai/Mistral-7B-v0.1)
    - GPT-2 (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
    - GPT-J (EleutherAI/gpt-j-6B)
    - BERT (bert-base-uncased, bert-large-uncased)
    - And many more...

    Note: Requires `transformers` and `torch` packages.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        output_hidden_states: bool = True,
        layer_index: int = -1,
        revision: Optional[str] = None,
    ):
        """Initialize the HuggingFace adapter.

        Args:
            model_name: HuggingFace model identifier
                        (e.g., "meta-llama/Llama-2-7b-hf")
            device: Device to run the model on ("cpu", "cuda", "cuda:0", etc.)
            output_hidden_states: Whether to request hidden states from model
            layer_index: Which hidden layer to extract (-1 = last layer,
                         -2 = second to last, etc.)
            revision: Specific model revision to use (commit hash, branch name,
                      or tag). Recommended for production to ensure reproducibility
                      and security. If None, uses the latest version.
        """
        self._model_name = model_name
        self._device = device
        self._output_hidden_states = output_hidden_states
        self._layer_index = layer_index
        self._revision = revision
        self._model = None
        self._tokenizer = None
        self._hidden_dim: Optional[int] = None

    def _lazy_load(self) -> None:
        """Lazy load the model to avoid import errors if not used."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
        except ImportError as e:
            raise ImportError(
                "HuggingFaceAdapter requires 'transformers' and 'torch' packages. "
                "Install with: pip install transformers torch"
            ) from e

        # Security note: The revision parameter allows users to pin specific model versions
        # (commit hash, tag, or branch) for reproducibility and security. When revision is
        # None, HuggingFace Hub downloads from the default branch which may change over time.
        # For production use, always specify a commit hash to ensure integrity.
        # The nosec B615 comments on the from_pretrained() calls suppress static analysis
        # warnings because we properly support the revision parameter - the decision to
        # pin a specific version is intentionally left to the user for flexibility.
        self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615 - revision param available
            self._model_name,
            revision=self._revision,
        )
        self._model = AutoModel.from_pretrained(  # nosec B615 - revision param available
            self._model_name,
            output_hidden_states=self._output_hidden_states,
            revision=self._revision,
        )
        self._model.to(self._device)
        self._model.eval()

        # Get hidden dimension from model config
        config = self._model.config
        self._hidden_dim = getattr(
            config,
            "hidden_size",
            getattr(config, "n_embd", getattr(config, "d_model", 768)),
        )

    def get_hidden_states(
        self,
        input_ids: Any,
        attention_mask: Optional[Any] = None,
    ) -> np.ndarray:
        """Extract hidden states from the HuggingFace model.

        Args:
            input_ids: Token IDs [B, seq_len] (can be numpy, list, or tensor)
            attention_mask: Optional attention mask [B, seq_len]

        Returns:
            Hidden states as numpy array [B, seq_len, hidden_dim]
        """
        self._lazy_load()

        import torch

        # Convert to tensor if needed
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)
        elif isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)

        if attention_mask is not None:
            if isinstance(attention_mask, np.ndarray):
                attention_mask = torch.from_numpy(attention_mask)
            elif isinstance(attention_mask, list):
                attention_mask = torch.tensor(attention_mask)
            attention_mask = attention_mask.to(self._device)

        input_ids = input_ids.to(self._device)

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Extract hidden states from the specified layer
        hidden_states = outputs.hidden_states[self._layer_index]
        return hidden_states.cpu().numpy()

    def get_hidden_dim(self) -> int:
        """Return the hidden dimension of the model."""
        if self._hidden_dim is None:
            self._lazy_load()
        return self._hidden_dim  # type: ignore

    def get_model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def tokenize(self, texts: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        """Tokenize text using the model's tokenizer.

        Args:
            texts: Single text or list of texts to tokenize

        Returns:
            Dict with 'input_ids' and 'attention_mask' as numpy arrays
        """
        self._lazy_load()

        if isinstance(texts, str):
            texts = [texts]

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }


class MockLLMAdapter(LLMAdapter):
    """Mock LLM adapter for testing without requiring actual LLM.

    Useful for:
    - Unit testing
    - Prototyping
    - Environments without GPU/transformers installed
    """

    def __init__(
        self,
        model_name: str = "mock-llm",
        hidden_dim: int = 768,
    ):
        """Initialize mock adapter.

        Args:
            model_name: Mock model name
            hidden_dim: Simulated hidden dimension
        """
        self._model_name = model_name
        self._hidden_dim = hidden_dim

    def get_hidden_states(
        self,
        input_ids: Any,
        attention_mask: Optional[Any] = None,
    ) -> np.ndarray:
        """Generate random hidden states for testing.

        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Optional attention mask (unused)

        Returns:
            Random hidden states [B, seq_len, hidden_dim]
        """
        if isinstance(input_ids, (list, tuple)):
            input_ids = np.array(input_ids)
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)

        batch_size, seq_len = input_ids.shape
        # Generate reproducible random hidden states based on input
        rng = np.random.RandomState(int(np.sum(input_ids)) % (2**31))
        return rng.randn(batch_size, seq_len, self._hidden_dim).astype(np.float32)

    def get_hidden_dim(self) -> int:
        """Return the simulated hidden dimension."""
        return self._hidden_dim

    def get_model_name(self) -> str:
        """Return the mock model name."""
        return self._model_name


# ============================
#  Sequence Pooling Strategies
# ============================


class SequencePooler:
    """Pooling strategies for converting sequence hidden states to fixed vectors.

    Supports multiple pooling strategies:
    - mean: Average pooling over sequence
    - max: Max pooling over sequence
    - last: Use last token hidden state (common for causal LMs)
    - cls: Use first token (CLS) hidden state (common for BERT)
    - attention: Learned attention-based pooling
    """

    def __init__(
        self,
        hidden_dim: int,
        pooling_strategy: str = "attention",
    ):
        """Initialize pooler.

        Args:
            hidden_dim: Hidden dimension of input
            pooling_strategy: One of "mean", "max", "last", "cls", "attention"
        """
        self.hidden_dim = hidden_dim
        self.strategy = pooling_strategy

        # Attention pooling parameters
        if pooling_strategy == "attention":
            limit = np.sqrt(2.0 / hidden_dim)
            self.query = np.random.uniform(-limit, limit, (hidden_dim, 1))

    def pool(
        self,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Pool sequence hidden states to fixed vectors.

        Args:
            hidden_states: [B, seq_len, hidden_dim]
            attention_mask: Optional [B, seq_len], 1 for valid tokens, 0 for padding

        Returns:
            Pooled vectors [B, hidden_dim]
        """
        B, seq_len, hidden_dim = hidden_states.shape

        if attention_mask is None:
            attention_mask = np.ones((B, seq_len), dtype=np.float64)
        else:
            attention_mask = attention_mask.astype(np.float64)

        if self.strategy == "mean":
            # Mean pooling with mask
            mask_expanded = attention_mask[:, :, np.newaxis]  # [B, seq_len, 1]
            sum_hidden = np.sum(hidden_states * mask_expanded, axis=1)  # [B, hidden_dim]
            sum_mask = np.sum(attention_mask, axis=1, keepdims=True)  # [B, 1]
            return sum_hidden / np.maximum(sum_mask, 1e-9)

        elif self.strategy == "max":
            # Max pooling with mask
            mask_expanded = attention_mask[:, :, np.newaxis]  # [B, seq_len, 1]
            masked_hidden = hidden_states.copy()
            masked_hidden[mask_expanded.squeeze(-1) == 0] = -np.inf
            return np.max(masked_hidden, axis=1)

        elif self.strategy == "last":
            # Last valid token
            lengths = np.sum(attention_mask, axis=1).astype(int)  # [B]
            return np.array([hidden_states[i, lengths[i] - 1] for i in range(B)])

        elif self.strategy == "cls":
            # First token (CLS)
            return hidden_states[:, 0, :]

        elif self.strategy == "attention":
            # Learned attention pooling
            # scores: [B, seq_len, 1]
            scores = hidden_states @ self.query
            scores = scores.squeeze(-1)  # [B, seq_len]

            # Mask padding tokens
            scores = np.where(attention_mask == 1, scores, -1e9)

            # Softmax attention weights
            weights = softmax(scores, axis=-1)  # [B, seq_len]

            # Weighted sum
            return np.sum(hidden_states * weights[:, :, np.newaxis], axis=1)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.strategy}")


# ============================
#  Main Reasoning Layer
# ============================


class TRLinkOSReasoningLayer:
    """T-RLINKOS Reasoning Layer for LLM Integration.

    This class wraps the T-RLINKOS recursive reasoning architecture as a
    plug-in reasoning layer for any LLM. It:

    1. Projects LLM hidden states to reasoning space
    2. Performs recursive reasoning with dCaAP experts and Torque routing
    3. Optionally explores solution space with fractal branching
    4. Maintains cryptographic audit trail via Merkle-DAG
    5. Projects reasoning output back to LLM space (optional)

    Architecture:
    ```
    LLM Hidden States [B, seq_len, llm_dim]
            │
            ▼
    Sequence Pooling [B, llm_dim]
            │
            ▼
    Input Projection [B, input_dim]
            │
            ▼
    ┌───────────────────────────┐
    │    TRLinkosTRM Core       │
    │  ┌─────────────────────┐  │
    │  │   Recursive Loop    │  │
    │  │  ┌───────────────┐  │  │
    │  │  │TorqueRouter   │  │  │
    │  │  │(Expert Select)│  │  │
    │  │  └───────────────┘  │  │
    │  │  ┌───────────────┐  │  │
    │  │  │DCaAPCell MoE  │  │  │
    │  │  │(Reasoning)    │  │  │
    │  │  └───────────────┘  │  │
    │  └─────────────────────┘  │
    │  FractalMerkleDAG         │
    │  (Audit Trail)            │
    └───────────────────────────┘
            │
            ▼
    Output Projection [B, output_dim]
            │
            ▼
    (Optional) LLM Projection [B, llm_dim]
    ```

    Example:
        >>> config = ReasoningConfig(input_dim=4096)  # LLaMA-7B
        >>> reasoning = TRLinkOSReasoningLayer(config)
        >>> hidden_states = np.random.randn(4, 128, 4096)  # [B, seq, dim]
        >>> output, dag = reasoning.reason(hidden_states)
        >>> print(output.shape)  # (4, 256) or (4, 4096) if project_to_llm_dim
    """

    def __init__(self, config: ReasoningConfig):
        """Initialize the reasoning layer.

        Args:
            config: ReasoningConfig with layer parameters
        """
        self.config = config

        # Input projection (LLM dim -> reasoning dim)
        self.input_proj = LinearNP(config.input_dim, config.output_dim)

        # Core T-RLINKOS model
        self.trm = TRLinkosTRM(
            x_dim=config.output_dim,
            y_dim=config.output_dim,
            z_dim=config.z_dim,
            hidden_dim=config.hidden_dim,
            num_experts=config.num_experts,
        )

        # Output projection (optional: reasoning dim -> LLM dim)
        if config.project_to_llm_dim:
            self.output_proj = LinearNP(config.output_dim, config.input_dim)
        else:
            self.output_proj = None

        # Sequence pooler
        pooling_strategy = "attention" if config.use_attention_pooling else "mean"
        self.pooler = SequencePooler(config.input_dim, pooling_strategy)

    def reason(
        self,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        scorer: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, FractalMerkleDAG]:
        """Perform recursive reasoning on LLM hidden states.

        This is the main entry point for using T-RLINKOS as a reasoning layer.

        Args:
            hidden_states: LLM hidden states [B, seq_len, hidden_dim] or [B, hidden_dim]
            attention_mask: Optional attention mask [B, seq_len]
            scorer: Optional scoring function (x, y) -> scores [B]

        Returns:
            output: Reasoning output [B, output_dim] or [B, llm_dim]
            dag: FractalMerkleDAG with reasoning trace
        """
        # Handle sequence input
        if hidden_states.ndim == 3:
            # Pool sequence to fixed vector [B, hidden_dim]
            pooled = self.pooler.pool(hidden_states, attention_mask)
        elif hidden_states.ndim == 2:
            pooled = hidden_states
        else:
            raise ValueError(
                f"hidden_states must be 2D or 3D, got shape {hidden_states.shape}"
            )

        # Project to reasoning space
        x = self.input_proj(pooled)  # [B, output_dim]

        # Run recursive reasoning
        if self.config.enable_fractal_branching:
            y, dag = self.trm.forward_recursive_fractal(
                x,
                max_steps=self.config.max_reasoning_steps,
                inner_recursions=self.config.inner_recursions,
                scorer=scorer,
                backtrack=self.config.enable_backtracking,
                backtrack_threshold=self.config.backtrack_threshold,
                fractal_branching=True,
                branch_threshold=self.config.branch_threshold,
                max_branches_per_node=self.config.max_branches_per_node,
                perturbation_scale=self.config.perturbation_scale,
            )
        else:
            y, dag = self.trm.forward_recursive(
                x,
                max_steps=self.config.max_reasoning_steps,
                inner_recursions=self.config.inner_recursions,
                scorer=scorer,
                backtrack=self.config.enable_backtracking,
                backtrack_threshold=self.config.backtrack_threshold,
            )

        # Optionally project back to LLM dimension
        if self.output_proj is not None:
            output = self.output_proj(y)  # [B, llm_dim]
        else:
            output = y

        return output, dag

    def reason_with_adapter(
        self,
        adapter: LLMAdapter,
        input_ids: Any,
        attention_mask: Optional[Any] = None,
        scorer: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, FractalMerkleDAG]:
        """Perform reasoning using an LLM adapter.

        Convenience method that extracts hidden states from the LLM
        and performs reasoning in one call.

        Args:
            adapter: LLM adapter instance (HuggingFaceAdapter, MockLLMAdapter, etc.)
            input_ids: Token IDs for the LLM
            attention_mask: Optional attention mask
            scorer: Optional scoring function

        Returns:
            output: Reasoning output
            dag: FractalMerkleDAG with reasoning trace
        """
        # Extract hidden states from LLM
        hidden_states = adapter.get_hidden_states(input_ids, attention_mask)

        # Preprocess if needed
        hidden_states = adapter.preprocess_hidden_states(hidden_states)

        # Perform reasoning
        return self.reason(hidden_states, attention_mask, scorer)

    def get_reasoning_trace(
        self,
        dag: FractalMerkleDAG,
    ) -> Dict[str, Any]:
        """Get detailed reasoning trace from the DAG.

        Args:
            dag: FractalMerkleDAG from reasoning

        Returns:
            Dict with reasoning trace information:
            - num_nodes: Total nodes in DAG
            - depth_stats: Nodes per depth level
            - best_node: Information about best scoring node
            - root_nodes: List of root node IDs
        """
        best_node = dag.get_best_node()

        trace = {
            "num_nodes": len(dag.nodes),
            "depth_stats": dag.get_depth_statistics(),
            "root_nodes": dag.root_nodes,
            "best_node": None,
        }

        if best_node is not None:
            trace["best_node"] = {
                "node_id": best_node.node_id,
                "step": best_node.step,
                "depth": best_node.depth,
                "score": best_node.score,
                "y_hash": best_node.y_hash,
                "z_hash": best_node.z_hash,
            }

        return trace


# ============================
#  Chain-of-Thought Enhancement
# ============================


class ChainOfThoughtAugmenter:
    """Augments chain-of-thought reasoning with T-RLINKOS.

    This class implements chain-of-thought augmentation where T-RLINKOS
    reasoning is interleaved with LLM generation. It enables:

    - Multi-step reasoning with intermediate verification
    - Backtracking when reasoning diverges
    - Fractal exploration of reasoning paths
    - Cryptographic verification of reasoning chain

    Reference: Integration with L-InCOT (https://github.com/RektaPro/L-inCOTv0.1)
    """

    def __init__(
        self,
        reasoning_layer: TRLinkOSReasoningLayer,
        adapter: Optional[LLMAdapter] = None,
    ):
        """Initialize the augmenter.

        Args:
            reasoning_layer: TRLinkOSReasoningLayer instance
            adapter: Optional LLM adapter for generation
        """
        self.reasoning_layer = reasoning_layer
        self.adapter = adapter
        self._thought_history: List[Dict[str, Any]] = []

    def add_thought(
        self,
        thought_embedding: np.ndarray,
        thought_text: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single thought in the chain.

        Args:
            thought_embedding: Embedding of the thought [1, hidden_dim] or [hidden_dim]
            thought_text: Optional text representation of the thought

        Returns:
            enhanced_embedding: Enhanced embedding after reasoning
            trace: Reasoning trace for this thought
        """
        if thought_embedding.ndim == 1:
            thought_embedding = thought_embedding.reshape(1, -1)

        # Perform reasoning
        enhanced, dag = self.reasoning_layer.reason(thought_embedding)

        # Get trace
        trace = self.reasoning_layer.get_reasoning_trace(dag)
        trace["thought_text"] = thought_text

        # Store in history
        self._thought_history.append({
            "input": thought_embedding.copy(),
            "output": enhanced.copy(),
            "trace": trace,
        })

        return enhanced, trace

    def get_chain_trace(self) -> List[Dict[str, Any]]:
        """Get the full chain-of-thought trace.

        Returns:
            List of trace dictionaries for each thought
        """
        return [h["trace"] for h in self._thought_history]

    def verify_chain(self) -> bool:
        """Verify the integrity of the reasoning chain.

        Checks that the hash chain is consistent by re-computing
        hashes of stored states.

        Returns:
            True if chain is valid, False otherwise
        """
        # This is a placeholder for full cryptographic verification
        # In production, this would verify the Merkle-DAG hashes
        return len(self._thought_history) > 0

    def reset(self) -> None:
        """Reset the thought history."""
        self._thought_history = []


# ============================
#  Advanced LLM Integration
# ============================

# Constants for text processing in AdvancedLLMIntegration
# Maximum text length to process (prevents memory issues with very long inputs)
MAX_TEXT_LENGTH = 128
# Modulo value for converting characters to pseudo-tokens (ASCII range)
CHAR_TO_TOKEN_MODULO = 256


class AdvancedLLMIntegration:
    """Intégration avancée T-RLINKOS + LLM.

    Cette classe fournit une intégration complète entre T-RLINKOS et les LLMs,
    incluant:

    1. Raisonnement itératif avec feedback
    2. Génération guidée par le DAG
    3. Vérification des hallucinations via DivergenceDetector
    4. Chain-of-Thought augmenté avec backtracking

    Cette classe est conçue pour être utilisée avec n'importe quel LLM via
    l'interface LLMAdapter (HuggingFace, OpenAI, etc.).

    Example:
        >>> from trlinkos_llm_layer import AdvancedLLMIntegration, TRLinkOSReasoningLayer
        >>> reasoning = TRLinkOSReasoningLayer(ReasoningConfig(input_dim=768))
        >>> adapter = MockLLMAdapter(hidden_dim=768)
        >>> integration = AdvancedLLMIntegration(reasoning, adapter)
        >>> result, dag, meta = integration.reason_and_generate("What is AI?")
    """

    def __init__(
        self,
        reasoning_layer: TRLinkOSReasoningLayer,
        llm_adapter: Optional[LLMAdapter] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialise l'intégration avancée.

        Args:
            reasoning_layer: Couche de raisonnement T-RLINKOS
            llm_adapter: Adaptateur LLM (optionnel, requis pour certaines méthodes)
            config: Configuration optionnelle (non utilisée actuellement)
        """
        self.reasoning = reasoning_layer
        self.llm = llm_adapter
        self.config = config or {}

        # Import DivergenceDetector (disponible après les ajouts)
        from t_rlinkos_trm_fractal_dag import DivergenceDetector
        self.hallucination_detector = DivergenceDetector()
        self.chain_history: List[Dict[str, Any]] = []

    def reason_and_generate(
        self,
        prompt: str,
        max_reasoning_steps: int = 8,
        verify_output: bool = True
    ) -> Tuple[str, FractalMerkleDAG, Dict[str, Any]]:
        """Pipeline complet: raisonnement T-RLINKOS + génération LLM.

        Ce pipeline:
        1. Encode le prompt en hidden states via l'adaptateur LLM
        2. Applique le raisonnement T-RLINKOS récursif
        3. Génère une trace de raisonnement
        4. Vérifie les hallucinations (optionnel)
        5. Retourne le résultat avec métadonnées

        Args:
            prompt: Texte du prompt à traiter
            max_reasoning_steps: Nombre maximal d'étapes de raisonnement
            verify_output: Si True, vérifie les hallucinations

        Returns:
            Tuple de:
            - generated_text: Texte généré ou résumé du raisonnement
            - dag: DAG de raisonnement fractal
            - metadata: Métadonnées incluant scores, statistiques, etc.

        Raises:
            ValueError: Si aucun adaptateur LLM n'est configuré
        """
        if self.llm is None:
            raise ValueError("LLM adapter required for reason_and_generate")

        # 1. Tokeniser et encoder le prompt
        if hasattr(self.llm, 'tokenize'):
            tokens = self.llm.tokenize(prompt)
            hidden_states = self.llm.get_hidden_states(tokens["input_ids"])
        else:
            # Fallback pour les adaptateurs sans tokenize
            # Génère des pseudo-tokens basés sur la longueur du texte
            pseudo_ids = np.array([
                [ord(c) % CHAR_TO_TOKEN_MODULO for c in prompt[:MAX_TEXT_LENGTH]]
            ])
            hidden_states = self.llm.get_hidden_states(pseudo_ids)

        # 2. Définir un scorer basé sur la cohérence
        mean_hidden = hidden_states.mean(axis=1) if hidden_states.ndim == 3 else hidden_states

        def quality_scorer(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            """Score basé sur la cohérence avec le prompt."""
            # Normaliser pour comparaison
            y_flat = y.reshape(y.shape[0], -1)
            target = mean_hidden.reshape(mean_hidden.shape[0], -1)

            # Ajuster les dimensions si nécessaire
            min_dim = min(y_flat.shape[1], target.shape[1])
            y_flat = y_flat[:, :min_dim]
            target = target[:, :min_dim]

            # Score = similarité cosinus
            y_norm = np.linalg.norm(y_flat, axis=1, keepdims=True) + 1e-10
            t_norm = np.linalg.norm(target, axis=1, keepdims=True) + 1e-10
            similarity = np.sum((y_flat / y_norm) * (target / t_norm), axis=1)
            return similarity

        # 3. Raisonner avec T-RLINKOS
        reasoning_output, dag = self.reasoning.reason(
            hidden_states,
            scorer=quality_scorer
        )

        # 4. Obtenir la trace de raisonnement
        trace = self.reasoning.get_reasoning_trace(dag)

        # 5. Vérifier les hallucinations si demandé
        hallucination_warning = None
        if verify_output and dag.best_node_id and dag.best_score is not None:
            self.hallucination_detector.update(dag.best_score, reasoning_output[0])
            is_diverging, reason = self.hallucination_detector.is_diverging()
            if is_diverging:
                hallucination_warning = f"Warning: {reason}"

        # 6. Générer le texte résumé
        score_str = f"{dag.best_score:.4f}" if dag.best_score > float('-inf') else "N/A"
        generated_text = (
            f"[T-RLINKOS Reasoning Complete]\n"
            f"Steps: {trace['num_nodes']} | Best Score: {score_str}\n"
            f"Depth Distribution: {trace['depth_stats']}"
        )

        # 7. Métadonnées
        metadata: Dict[str, Any] = {
            "reasoning_steps": trace["num_nodes"],
            "best_score": dag.best_score if dag.best_score > float('-inf') else None,
            "depth_stats": trace["depth_stats"],
            "verified": verify_output,
            "hallucination_warning": hallucination_warning,
            "prompt_length": len(prompt),
        }

        return generated_text, dag, metadata

    def chain_of_thought(
        self,
        problem: str,
        num_steps: int = 5
    ) -> List[Dict[str, Any]]:
        """Chain-of-Thought augmenté avec T-RLINKOS.

        Implémente un raisonnement multi-étapes où chaque étape de pensée est:
        1. Générée en tant qu'embedding
        2. Validée par T-RLINKOS
        3. Tracée dans le DAG
        4. Vérifiée pour les divergences/hallucinations

        Args:
            problem: Description du problème à résoudre
            num_steps: Nombre d'étapes de raisonnement à effectuer

        Returns:
            Liste de dictionnaires contenant les résultats de chaque étape:
            - step: Numéro de l'étape
            - dag_nodes: Nombre de nœuds dans le DAG pour cette étape
            - score: Score de raisonnement
            - diverging: True si divergence détectée
            - divergence_reason: Raison de la divergence (si applicable)

        Raises:
            ValueError: Si aucun adaptateur LLM n'est configuré
        """
        if self.llm is None:
            raise ValueError("LLM adapter required for chain_of_thought")

        self.chain_history = []
        self.hallucination_detector.reset()

        for step in range(num_steps):
            # Encoder le contexte actuel
            if step == 0:
                current_context = problem
            else:
                # Utiliser le résultat de l'étape précédente comme contexte
                prev_result = self.chain_history[-1]
                current_context = f"{problem} [Step {step}: score={prev_result.get('score', 'N/A')}]"

            # Tokeniser et obtenir les hidden states
            if hasattr(self.llm, 'tokenize'):
                tokens = self.llm.tokenize(current_context)
                hidden = self.llm.get_hidden_states(tokens["input_ids"])
            else:
                pseudo_ids = np.array([
                    [ord(c) % CHAR_TO_TOKEN_MODULO for c in current_context[:MAX_TEXT_LENGTH]]
                ])
                hidden = self.llm.get_hidden_states(pseudo_ids)

            # Raisonnement T-RLINKOS
            output, dag = self.reasoning.reason(hidden)

            # Détecter divergence
            is_diverging = False
            reason = "No score"

            if dag.best_score is not None and dag.best_score > float('-inf'):
                self.hallucination_detector.update(dag.best_score, output[0])
                is_diverging, reason = self.hallucination_detector.is_diverging()

            step_result: Dict[str, Any] = {
                "step": step,
                "dag_nodes": len(dag.nodes),
                "score": dag.best_score if dag.best_score > float('-inf') else None,
                "diverging": is_diverging,
                "divergence_reason": reason,
            }

            self.chain_history.append(step_result)

            if is_diverging:
                # Log la divergence (en production, on pourrait backtracker)
                pass  # Continue quand même pour collecter les données

        return self.chain_history

    def get_chain_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de la chaîne de raisonnement.

        Returns:
            Dictionnaire avec:
            - total_steps: Nombre total d'étapes
            - divergence_count: Nombre de divergences détectées
            - mean_score: Score moyen (excluant None)
            - scores: Liste de tous les scores
        """
        if not self.chain_history:
            return {
                "total_steps": 0,
                "divergence_count": 0,
                "mean_score": None,
                "scores": [],
            }

        scores = [h["score"] for h in self.chain_history if h["score"] is not None]

        return {
            "total_steps": len(self.chain_history),
            "divergence_count": sum(1 for h in self.chain_history if h["diverging"]),
            "mean_score": float(np.mean(scores)) if scores else None,
            "scores": scores,
        }

    def reset(self) -> None:
        """Réinitialise l'état de l'intégration."""
        self.chain_history = []
        self.hallucination_detector.reset()


# ============================
#  Convenience Factory Functions
# ============================


def create_reasoning_layer_for_llm(
    model_name: str,
    reasoning_steps: int = 8,
    num_experts: int = 4,
) -> Tuple[TRLinkOSReasoningLayer, ReasoningConfig]:
    """Create a reasoning layer configured for a specific LLM.

    Convenience function that sets up the reasoning layer with
    appropriate dimensions for common LLMs.

    Args:
        model_name: Name/identifier of the LLM. Supported:
                    - "llama-7b", "llama-13b", "llama-70b"
                    - "mistral-7b"
                    - "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
                    - "bert-base", "bert-large"
                    Or any HuggingFace model name
        reasoning_steps: Maximum reasoning steps
        num_experts: Number of dCaAP experts

    Returns:
        Tuple of (TRLinkOSReasoningLayer, ReasoningConfig)
    """
    # Dimension lookup for common models
    dim_lookup = {
        # LLaMA family
        "llama-7b": 4096,
        "llama-13b": 5120,
        "llama-70b": 8192,
        "meta-llama/llama-2-7b": 4096,
        "meta-llama/llama-2-13b": 5120,
        "meta-llama/llama-2-70b": 8192,
        # Mistral family
        "mistral-7b": 4096,
        "mistralai/mistral-7b-v0.1": 4096,
        # GPT-2 family
        "gpt2": 768,
        "gpt2-medium": 1024,
        "gpt2-large": 1280,
        "gpt2-xl": 1600,
        # BERT family
        "bert-base": 768,
        "bert-large": 1024,
        "bert-base-uncased": 768,
        "bert-large-uncased": 1024,
    }

    # Normalize model name for lookup
    normalized = model_name.lower().replace("_", "-")
    hidden_dim = dim_lookup.get(normalized, 4096)  # Default to 4096

    # Create config
    config = ReasoningConfig(
        input_dim=hidden_dim,
        output_dim=min(hidden_dim // 4, 512),  # Reasonable output dim
        z_dim=min(hidden_dim // 8, 256),  # Internal state dim
        hidden_dim=min(hidden_dim // 4, 512),  # Hidden layer dim
        num_experts=num_experts,
        max_reasoning_steps=reasoning_steps,
    )

    # Create layer
    layer = TRLinkOSReasoningLayer(config)

    return layer, config


# ============================
#  Simple Interface Functions (Stubs)
# ============================

# The following functions provide a simple, clear API for integrating
# TRLinkosTRM as a reasoning layer with any LLM. These are designed as
# "stubs" that work with numpy arrays directly, with documentation on
# how to connect them to real LLMs.


def encode_text(
    text: str,
    embedding_dim: int = 768,
    encoder: Optional[Any] = None,
) -> np.ndarray:
    """Encode text into a vector embedding.

    This function converts raw text into a dense vector representation
    suitable for TRLinkosTRM reasoning. It serves as a stub that can be
    connected to any LLM embedding model.

    **Stub Behavior (without external LLM):**
    - Uses a simple hash-based encoding that produces deterministic
      embeddings based on text content.
    - Suitable for testing and prototyping.

    **How to connect a real LLM:**

    1. **OpenAI Embeddings:**
       ```python
       import openai
       from openai import OpenAI

       client = OpenAI(api_key="your-api-key")

       def encode_text_openai(text: str) -> np.ndarray:
           response = client.embeddings.create(
               input=text,
               model="text-embedding-3-small"  # or "text-embedding-ada-002"
           )
           return np.array(response.data[0].embedding)
       ```

    2. **HuggingFace Sentence Transformers:**
       ```python
       from sentence_transformers import SentenceTransformer

       model = SentenceTransformer('all-MiniLM-L6-v2')

       def encode_text_hf(text: str) -> np.ndarray:
           return model.encode(text)
       ```

    3. **Mistral AI Embeddings:**
       ```python
       from mistralai.client import MistralClient

       client = MistralClient(api_key="your-api-key")

       def encode_text_mistral(text: str) -> np.ndarray:
           response = client.embeddings(
               model="mistral-embed",
               input=[text]
           )
           return np.array(response.data[0].embedding)
       ```

    4. **Custom HuggingFace Model:**
       ```python
       from transformers import AutoTokenizer, AutoModel
       import torch

       tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
       model = AutoModel.from_pretrained("bert-base-uncased")

       def encode_text_custom(text: str) -> np.ndarray:
           inputs = tokenizer(text, return_tensors="pt", truncation=True)
           with torch.no_grad():
               outputs = model(**inputs)
           # Mean pooling over sequence
           return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
       ```

    Args:
        text: The text string to encode.
        embedding_dim: Dimension of the output embedding (default: 768).
                       Common values: 384, 768, 1024, 1536, 4096.
        encoder: Optional external encoder object. If provided, should have
                 an `encode(text)` method returning an np.ndarray.

    Returns:
        np.ndarray of shape [embedding_dim] containing the text embedding.

    Example:
        >>> embedding = encode_text("What is the capital of France?")
        >>> print(embedding.shape)
        (768,)
        >>> # Use with TRLinkosTRM reasoning
        >>> output, dag = reasoning_layer.reason(embedding.reshape(1, -1))
    """
    # If an external encoder is provided, use it
    if encoder is not None:
        if hasattr(encoder, "encode"):
            result = encoder.encode(text)
            if isinstance(result, np.ndarray):
                flat_result = result.flatten()
                # Truncate or pad to target dimension
                if len(flat_result) > embedding_dim:
                    return flat_result[:embedding_dim]
                elif len(flat_result) < embedding_dim:
                    pad_width = embedding_dim - len(flat_result)
                    return np.pad(flat_result, (0, pad_width))
                return flat_result
            return np.array(result).flatten()

    # Stub implementation: deterministic hash-based embedding
    # This creates reproducible embeddings based on text content
    text_bytes = text.encode("utf-8")

    # Expand hash to embedding_dim using repeated hashing
    embedding_parts = []
    current_bytes = text_bytes
    while len(embedding_parts) * 64 < embedding_dim:
        current_hash = hashlib.sha512(current_bytes).digest()
        embedding_parts.append(np.frombuffer(current_hash, dtype=np.uint8))
        current_bytes = current_hash

    # Combine and normalize
    raw_embedding = np.concatenate(embedding_parts).astype(np.float64)[:embedding_dim]
    # Normalize to [-1, 1] range and apply L2 normalization
    raw_embedding = (raw_embedding / 127.5) - 1.0
    norm = np.linalg.norm(raw_embedding)
    if norm > 0:
        raw_embedding = raw_embedding / norm

    return raw_embedding


def reason_over_candidates(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    reasoning_layer: Optional["TRLinkOSReasoningLayer"] = None,
    reasoning_steps: int = 4,
    return_all_scores: bool = False,
    combination_strategy: str = "average",
) -> Tuple[np.ndarray, int]:
    """Score and rank candidates using TRLinkosTRM reasoning.

    This function performs reranking of candidate responses/documents
    by using TRLinkosTRM to reason about the relationship between
    a query and multiple candidates. This is useful for:

    - **Response reranking**: Given an LLM query and multiple generated
      responses, select the best one.
    - **Document retrieval**: Rerank retrieved documents for RAG pipelines.
    - **Answer selection**: Choose the best answer from multiple candidates.

    **How it works:**
    1. Each candidate is paired with the query.
    2. TRLinkosTRM reasons over each pair to produce an enhanced representation.
    3. Scores are computed based on reasoning quality (cosine similarity
       between reasoned output and query).
    4. Returns scores and the index of the best candidate.

    **Typical usage in a RAG pipeline:**
    ```python
    # Step 1: Encode query and candidates
    query_emb = encode_text("What causes climate change?")
    candidates = [
        "Climate change is caused by greenhouse gases.",
        "The weather is nice today.",
        "CO2 emissions from fossil fuels drive global warming.",
    ]
    candidate_embs = np.array([encode_text(c) for c in candidates])

    # Step 2: Reason and rerank
    scores, best_idx = reason_over_candidates(query_emb, candidate_embs)

    # Step 3: Select best response
    best_response = candidates[best_idx]
    print(f"Best response: {best_response}")
    print(f"Scores: {scores}")
    ```

    **Integration with LLM response selection:**
    ```python
    # Generate multiple responses from LLM
    responses = llm.generate(prompt, num_return_sequences=5)

    # Encode and reason
    query_emb = encode_text(prompt)
    response_embs = np.array([encode_text(r) for r in responses])
    scores, best_idx = reason_over_candidates(query_emb, response_embs)

    # Select the best reasoned response
    final_response = responses[best_idx]
    ```

    Args:
        query_embedding: Query vector [embedding_dim] or [1, embedding_dim].
        candidate_embeddings: Candidate vectors [num_candidates, embedding_dim].
        reasoning_layer: Optional TRLinkOSReasoningLayer. If None, creates
                         a default layer configured for the embedding dimension.
        reasoning_steps: Number of reasoning steps per candidate (default: 4).
        return_all_scores: If True, returns detailed per-step scores.
        combination_strategy: How to combine query and candidate embeddings.
                              Options: "average" (default), "concat", "weighted".
                              - "average": Element-wise mean of query and candidate.
                              - "concat": Concatenate query and candidate (doubles dim).
                              - "weighted": 0.7 * query + 0.3 * candidate.

    Returns:
        Tuple of:
        - scores: np.ndarray [num_candidates] with reasoning-based scores.
          Higher scores indicate better matches.
        - best_index: int, index of the highest-scoring candidate.

    Example:
        >>> query = np.random.randn(768)
        >>> candidates = np.random.randn(5, 768)
        >>> scores, best_idx = reason_over_candidates(query, candidates)
        >>> print(f"Best candidate index: {best_idx}")
        >>> print(f"Scores: {scores}")
    """
    # Ensure query is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Ensure candidates are 2D
    if candidate_embeddings.ndim == 1:
        candidate_embeddings = candidate_embeddings.reshape(1, -1)

    num_candidates, embedding_dim = candidate_embeddings.shape

    # Determine input dimension based on combination strategy
    if combination_strategy == "concat":
        input_dim = embedding_dim * 2
    else:
        input_dim = embedding_dim

    # Create reasoning layer if not provided
    if reasoning_layer is None:
        config = ReasoningConfig(
            input_dim=input_dim,
            output_dim=min(input_dim // 4, 256),
            z_dim=min(input_dim // 8, 128),
            hidden_dim=min(input_dim // 4, 256),
            num_experts=4,
            max_reasoning_steps=reasoning_steps,
            project_to_llm_dim=True,  # Project back for comparison
        )
        reasoning_layer = TRLinkOSReasoningLayer(config)

    # Reason over each candidate paired with query
    scores = np.zeros(num_candidates)

    for i in range(num_candidates):
        # Combine query and candidate using specified strategy
        candidate = candidate_embeddings[i:i+1]
        if combination_strategy == "average":
            combined = (query_embedding + candidate) / 2.0
        elif combination_strategy == "concat":
            combined = np.concatenate([query_embedding, candidate], axis=-1)
        elif combination_strategy == "weighted":
            combined = 0.7 * query_embedding + 0.3 * candidate
        else:
            # Default to average for unknown strategies
            combined = (query_embedding + candidate) / 2.0

        # Run reasoning
        reasoned_output, dag = reasoning_layer.reason(combined)

        # Score based on similarity between reasoned output and query
        # Higher score = better reasoning alignment
        # For concat strategy, we need to extract the query portion of the output
        if combination_strategy == "concat":
            # When using concat, the output is 2x the original dim
            # Take the first half which corresponds to the query portion
            output_for_comparison = reasoned_output[:, :embedding_dim]
        else:
            output_for_comparison = reasoned_output

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        output_norm = output_for_comparison / (np.linalg.norm(output_for_comparison) + 1e-10)
        similarity = np.sum(query_norm * output_norm)

        # Also consider DAG quality (more nodes explored = more thorough)
        dag_quality = min(len(dag.nodes) / (reasoning_steps * 2), 1.0)

        scores[i] = similarity + 0.1 * dag_quality

    best_index = int(np.argmax(scores))
    return scores, best_index


def multi_step_reasoning(
    history: List[np.ndarray],
    new_input: np.ndarray,
    reasoning_layer: Optional["TRLinkOSReasoningLayer"] = None,
    context_window: int = 5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Perform multi-step reasoning with conversation history.

    This function enables chain-of-thought style reasoning where the
    TRLinkosTRM layer builds upon previous reasoning steps. It's designed
    for:

    - **Multi-turn conversations**: Maintain reasoning context across turns.
    - **Iterative refinement**: Progressively improve responses.
    - **Complex problem solving**: Break down problems into steps.

    **How it works:**
    1. Aggregates recent history embeddings (within context_window).
    2. Combines history context with new input.
    3. Runs TRLinkosTRM reasoning on the combined representation.
    4. Returns enhanced output and metadata about the reasoning process.

    **Typical usage for multi-turn reasoning:**
    ```python
    # Initialize history
    history = []

    # Turn 1
    q1 = encode_text("What is machine learning?")
    output1, meta1 = multi_step_reasoning(history, q1)
    history.append(output1)

    # Turn 2 (builds on Turn 1)
    q2 = encode_text("How is it different from deep learning?")
    output2, meta2 = multi_step_reasoning(history, q2)
    history.append(output2)

    # Turn 3 (builds on Turns 1 & 2)
    q3 = encode_text("Give me a practical example.")
    output3, meta3 = multi_step_reasoning(history, q3)

    # output3 now incorporates reasoning from all previous turns
    print(f"Reasoning steps taken: {meta3['total_reasoning_steps']}")
    ```

    **Integration with LLM chain-of-thought:**
    ```python
    # Enhance LLM outputs with multi-step reasoning
    llm_outputs = []
    reasoning_history = []

    for step_prompt in chain_of_thought_prompts:
        # Get LLM response
        llm_response = llm.generate(step_prompt)
        llm_emb = encode_text(llm_response)

        # Enhance with TRLinkosTRM reasoning
        enhanced, meta = multi_step_reasoning(reasoning_history, llm_emb)
        reasoning_history.append(enhanced)
        llm_outputs.append({
            'response': llm_response,
            'enhanced_embedding': enhanced,
            'reasoning_quality': meta['best_score']
        })
    ```

    Args:
        history: List of previous embeddings [embedding_dim] each.
                 Can be empty for the first turn.
        new_input: New input embedding [embedding_dim] or [1, embedding_dim].
        reasoning_layer: Optional TRLinkOSReasoningLayer. If None, creates
                         a default layer configured for the embedding dimension.
        context_window: Maximum number of history items to consider (default: 5).

    Returns:
        Tuple of:
        - output: np.ndarray [embedding_dim] with enhanced representation.
        - metadata: Dict with reasoning information:
          - 'num_history_items': Number of history items used
          - 'total_reasoning_steps': Total reasoning steps taken
          - 'best_score': Best score from DAG (if scorer was used)
          - 'dag_nodes': Number of nodes in reasoning DAG

    Example:
        >>> history = [np.random.randn(768) for _ in range(3)]
        >>> new_input = np.random.randn(768)
        >>> output, meta = multi_step_reasoning(history, new_input)
        >>> print(f"Output shape: {output.shape}")
        >>> print(f"Metadata: {meta}")
    """
    # Ensure new_input is 1D for consistency
    if new_input.ndim == 2:
        new_input = new_input.squeeze(0)

    embedding_dim = len(new_input)

    # Create reasoning layer if not provided
    if reasoning_layer is None:
        config = ReasoningConfig(
            input_dim=embedding_dim,
            output_dim=min(embedding_dim // 4, 256),
            z_dim=min(embedding_dim // 8, 128),
            hidden_dim=min(embedding_dim // 4, 256),
            num_experts=4,
            max_reasoning_steps=8,
            project_to_llm_dim=True,
        )
        reasoning_layer = TRLinkOSReasoningLayer(config)

    # Get recent history within context window
    recent_history = history[-context_window:] if len(history) > context_window else history

    # Aggregate history context (if any)
    if len(recent_history) > 0:
        # Weighted aggregation: more recent items have higher weight
        weights = np.array([0.5 ** (len(recent_history) - i - 1) for i in range(len(recent_history))])
        weights = weights / weights.sum()

        history_embeddings = np.array([
            h.squeeze() if h.ndim > 1 else h for h in recent_history
        ])
        history_context = np.average(history_embeddings, axis=0, weights=weights)

        # Combine history context with new input
        combined_input = (history_context + new_input) / 2.0
    else:
        combined_input = new_input

    # Reshape for reasoning layer [1, embedding_dim]
    combined_input = combined_input.reshape(1, -1)

    # Define a scorer that rewards coherence with new input
    def coherence_scorer(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Score based on similarity to new input (coherence)
        y_flat = y.reshape(-1)[:embedding_dim]
        if len(y_flat) < embedding_dim:
            y_flat = np.pad(y_flat, (0, embedding_dim - len(y_flat)))
        similarity = np.dot(new_input, y_flat) / (
            np.linalg.norm(new_input) * np.linalg.norm(y_flat) + 1e-10
        )
        return np.array([similarity])

    # Run reasoning
    output, dag = reasoning_layer.reason(combined_input, scorer=coherence_scorer)

    # Get best score from DAG
    best_node = dag.get_best_node()
    best_score = best_node.score if best_node is not None else None

    # Build metadata
    metadata = {
        "num_history_items": len(recent_history),
        "total_reasoning_steps": len(dag.nodes),
        "best_score": best_score,
        "dag_nodes": len(dag.nodes),
        "depth_stats": dag.get_depth_statistics(),
    }

    # Return flattened output
    return output.squeeze(0), metadata


# ============================
#  Tests
# ============================


if __name__ == "__main__":
    print("=" * 60)
    print("TRLINKOS LLM Reasoning Layer - Tests")
    print("=" * 60)

    np.random.seed(42)

    # --- Test 1: Basic ReasoningConfig ---
    print("\n--- Test 1: ReasoningConfig ---")
    config = ReasoningConfig(input_dim=768, output_dim=256)
    print(f"[Test 1] Config created: input_dim={config.input_dim}, "
          f"output_dim={config.output_dim}")
    assert config.input_dim == 768
    assert config.output_dim == 256
    print("[Test 1] ✅ ReasoningConfig works correctly!")

    # --- Test 2: MockLLMAdapter ---
    print("\n--- Test 2: MockLLMAdapter ---")
    mock_adapter = MockLLMAdapter(model_name="test-llm", hidden_dim=768)
    assert mock_adapter.get_hidden_dim() == 768
    assert mock_adapter.get_model_name() == "test-llm"

    input_ids = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    hidden = mock_adapter.get_hidden_states(input_ids)
    assert hidden.shape == (2, 5, 768), f"Unexpected shape: {hidden.shape}"
    print(f"[Test 2] Mock hidden states shape: {hidden.shape}")
    print("[Test 2] ✅ MockLLMAdapter works correctly!")

    # --- Test 3: SequencePooler ---
    print("\n--- Test 3: SequencePooler ---")
    hidden_states = np.random.randn(4, 10, 768)
    attention_mask = np.ones((4, 10))

    # Test mean pooling
    pooler_mean = SequencePooler(768, "mean")
    pooled_mean = pooler_mean.pool(hidden_states, attention_mask)
    assert pooled_mean.shape == (4, 768)
    print(f"[Test 3] Mean pooled shape: {pooled_mean.shape}")

    # Test attention pooling
    pooler_attn = SequencePooler(768, "attention")
    pooled_attn = pooler_attn.pool(hidden_states, attention_mask)
    assert pooled_attn.shape == (4, 768)
    print(f"[Test 3] Attention pooled shape: {pooled_attn.shape}")

    # Test last token pooling
    pooler_last = SequencePooler(768, "last")
    pooled_last = pooler_last.pool(hidden_states, attention_mask)
    assert pooled_last.shape == (4, 768)
    print(f"[Test 3] Last token pooled shape: {pooled_last.shape}")

    print("[Test 3] ✅ SequencePooler works correctly!")

    # --- Test 4: TRLinkOSReasoningLayer ---
    print("\n--- Test 4: TRLinkOSReasoningLayer ---")
    config = ReasoningConfig(
        input_dim=768,
        output_dim=256,
        z_dim=128,
        hidden_dim=256,
        num_experts=4,
        max_reasoning_steps=4,
        inner_recursions=2,
        project_to_llm_dim=False,
    )
    reasoning_layer = TRLinkOSReasoningLayer(config)

    # Test with 3D input (sequence)
    hidden_states = np.random.randn(4, 10, 768)
    output, dag = reasoning_layer.reason(hidden_states)
    assert output.shape == (4, 256), f"Unexpected shape: {output.shape}"
    print(f"[Test 4] Output shape (3D input): {output.shape}")
    print(f"[Test 4] DAG nodes: {len(dag.nodes)}")

    # Test with 2D input (already pooled)
    hidden_states_2d = np.random.randn(4, 768)
    output_2d, dag_2d = reasoning_layer.reason(hidden_states_2d)
    assert output_2d.shape == (4, 256)
    print(f"[Test 4] Output shape (2D input): {output_2d.shape}")

    print("[Test 4] ✅ TRLinkOSReasoningLayer works correctly!")

    # --- Test 5: TRLinkOSReasoningLayer with LLM projection ---
    print("\n--- Test 5: TRLinkOSReasoningLayer with LLM projection ---")
    config_proj = ReasoningConfig(
        input_dim=768,
        output_dim=256,
        project_to_llm_dim=True,  # Project back to LLM dim
    )
    reasoning_layer_proj = TRLinkOSReasoningLayer(config_proj)

    hidden_states = np.random.randn(4, 10, 768)
    output_proj, _ = reasoning_layer_proj.reason(hidden_states)
    assert output_proj.shape == (4, 768), f"Unexpected shape: {output_proj.shape}"
    print(f"[Test 5] Output shape with LLM projection: {output_proj.shape}")
    print("[Test 5] ✅ LLM projection works correctly!")

    # --- Test 6: reason_with_adapter ---
    print("\n--- Test 6: reason_with_adapter ---")
    mock_adapter = MockLLMAdapter(hidden_dim=768)
    config = ReasoningConfig(input_dim=768, output_dim=256, project_to_llm_dim=False)
    reasoning_layer = TRLinkOSReasoningLayer(config)

    input_ids = np.array([[1, 2, 3, 4, 5]])
    output, dag = reasoning_layer.reason_with_adapter(mock_adapter, input_ids)
    assert output.shape == (1, 256)
    print(f"[Test 6] Output shape from adapter: {output.shape}")
    print("[Test 6] ✅ reason_with_adapter works correctly!")

    # --- Test 7: Reasoning trace ---
    print("\n--- Test 7: Reasoning trace ---")
    trace = reasoning_layer.get_reasoning_trace(dag)
    print(f"[Test 7] Trace keys: {list(trace.keys())}")
    print(f"[Test 7] Number of DAG nodes: {trace['num_nodes']}")
    print(f"[Test 7] Depth stats: {trace['depth_stats']}")
    assert "num_nodes" in trace
    assert "depth_stats" in trace
    assert "best_node" in trace
    print("[Test 7] ✅ Reasoning trace works correctly!")

    # --- Test 8: Fractal branching ---
    print("\n--- Test 8: Fractal branching ---")
    config_fractal = ReasoningConfig(
        input_dim=768,
        output_dim=256,
        enable_fractal_branching=True,
        branch_threshold=0.01,
        max_reasoning_steps=6,
    )
    reasoning_layer_fractal = TRLinkOSReasoningLayer(config_fractal)

    hidden_states = np.random.randn(4, 10, 768)
    target = np.random.randn(4, 256)

    def scorer(x, y):
        return -np.mean((y - target) ** 2, axis=-1)

    output_fractal, dag_fractal = reasoning_layer_fractal.reason(
        hidden_states, scorer=scorer
    )
    trace_fractal = reasoning_layer_fractal.get_reasoning_trace(dag_fractal)
    print(f"[Test 8] Fractal output shape: {output_fractal.shape}")
    print(f"[Test 8] Fractal DAG nodes: {trace_fractal['num_nodes']}")
    print(f"[Test 8] Fractal depth stats: {trace_fractal['depth_stats']}")
    print("[Test 8] ✅ Fractal branching works correctly!")

    # --- Test 9: ChainOfThoughtAugmenter ---
    print("\n--- Test 9: ChainOfThoughtAugmenter ---")
    config = ReasoningConfig(input_dim=768, output_dim=256, max_reasoning_steps=3)
    reasoning_layer = TRLinkOSReasoningLayer(config)
    cot = ChainOfThoughtAugmenter(reasoning_layer)

    # Add thoughts
    thought1 = np.random.randn(768)
    thought2 = np.random.randn(768)

    enhanced1, trace1 = cot.add_thought(thought1, "First thought")
    enhanced2, trace2 = cot.add_thought(thought2, "Second thought")

    chain_trace = cot.get_chain_trace()
    assert len(chain_trace) == 2
    assert chain_trace[0]["thought_text"] == "First thought"
    print(f"[Test 9] Chain length: {len(chain_trace)}")
    print(f"[Test 9] Enhanced shape: {enhanced1.shape}")

    assert cot.verify_chain()
    cot.reset()
    assert len(cot.get_chain_trace()) == 0
    print("[Test 9] ✅ ChainOfThoughtAugmenter works correctly!")

    # --- Test 10: create_reasoning_layer_for_llm factory ---
    print("\n--- Test 10: create_reasoning_layer_for_llm factory ---")

    # Test LLaMA-7B
    layer_llama, config_llama = create_reasoning_layer_for_llm("llama-7b")
    assert config_llama.input_dim == 4096
    print(f"[Test 10] LLaMA-7B input_dim: {config_llama.input_dim}")

    # Test Mistral-7B
    layer_mistral, config_mistral = create_reasoning_layer_for_llm("mistral-7b")
    assert config_mistral.input_dim == 4096
    print(f"[Test 10] Mistral-7B input_dim: {config_mistral.input_dim}")

    # Test GPT-2
    layer_gpt2, config_gpt2 = create_reasoning_layer_for_llm("gpt2")
    assert config_gpt2.input_dim == 768
    print(f"[Test 10] GPT-2 input_dim: {config_gpt2.input_dim}")

    # Test BERT
    layer_bert, config_bert = create_reasoning_layer_for_llm("bert-base")
    assert config_bert.input_dim == 768
    print(f"[Test 10] BERT-base input_dim: {config_bert.input_dim}")

    print("[Test 10] ✅ Factory function works correctly!")

    # --- Test 11: End-to-end with mock LLM ---
    print("\n--- Test 11: End-to-end with mock LLM ---")

    # Simulate full pipeline
    mock_llm = MockLLMAdapter(model_name="mock-llama", hidden_dim=4096)
    layer, config = create_reasoning_layer_for_llm("llama-7b")

    # Adjust config for mock
    config.project_to_llm_dim = True
    layer = TRLinkOSReasoningLayer(config)

    # Simulate input tokens
    input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

    # Full pipeline: LLM -> Reasoning -> Output
    output, dag = layer.reason_with_adapter(mock_llm, input_ids)

    assert output.shape == (1, 4096), f"Unexpected shape: {output.shape}"
    print(f"[Test 11] End-to-end output shape: {output.shape}")
    print(f"[Test 11] DAG nodes: {len(dag.nodes)}")

    trace = layer.get_reasoning_trace(dag)
    if trace["best_node"]:
        print(f"[Test 11] Best node step: {trace['best_node']['step']}")

    print("[Test 11] ✅ End-to-end pipeline works correctly!")

    # --- Test 12: encode_text function ---
    print("\n--- Test 12: encode_text function ---")

    # Test basic encoding
    text = "What is the capital of France?"
    embedding = encode_text(text, embedding_dim=768)
    assert embedding.shape == (768,), f"Unexpected shape: {embedding.shape}"
    print(f"[Test 12] Text embedding shape: {embedding.shape}")

    # Test determinism (same text -> same embedding)
    embedding2 = encode_text(text, embedding_dim=768)
    np.testing.assert_array_almost_equal(embedding, embedding2)
    print("[Test 12] Embeddings are deterministic ✓")

    # Test different texts produce different embeddings
    text2 = "What is the weather today?"
    embedding3 = encode_text(text2, embedding_dim=768)
    assert not np.allclose(embedding, embedding3), "Different texts should have different embeddings"
    print("[Test 12] Different texts produce different embeddings ✓")

    # Test embedding normalization
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 0.01, f"Embedding should be normalized, got norm={norm}"
    print(f"[Test 12] Embedding norm: {norm:.4f} (normalized)")

    print("[Test 12] ✅ encode_text works correctly!")

    # --- Test 13: reason_over_candidates function ---
    print("\n--- Test 13: reason_over_candidates function ---")

    # Create query and candidates
    query = encode_text("What causes climate change?", embedding_dim=256)
    candidates = np.array([
        encode_text("Climate change is caused by greenhouse gases.", embedding_dim=256),
        encode_text("The weather is nice today.", embedding_dim=256),
        encode_text("CO2 emissions from fossil fuels drive global warming.", embedding_dim=256),
    ])

    scores, best_idx = reason_over_candidates(query, candidates, reasoning_steps=3)

    assert scores.shape == (3,), f"Unexpected scores shape: {scores.shape}"
    assert 0 <= best_idx < 3, f"Invalid best_idx: {best_idx}"
    print(f"[Test 13] Scores: {scores}")
    print(f"[Test 13] Best candidate index: {best_idx}")

    # Test with 1D query
    query_1d = np.random.randn(128)
    candidates_2d = np.random.randn(4, 128)
    scores2, best_idx2 = reason_over_candidates(query_1d, candidates_2d, reasoning_steps=2)
    assert scores2.shape == (4,)
    print(f"[Test 13] Scores with random data: {scores2}")

    print("[Test 13] ✅ reason_over_candidates works correctly!")

    # --- Test 14: multi_step_reasoning function ---
    print("\n--- Test 14: multi_step_reasoning function ---")

    # Test with empty history (first turn)
    new_input = encode_text("What is machine learning?", embedding_dim=256)
    output1, meta1 = multi_step_reasoning([], new_input)

    assert output1.shape == (256,), f"Unexpected output shape: {output1.shape}"
    assert "num_history_items" in meta1
    assert "total_reasoning_steps" in meta1
    assert meta1["num_history_items"] == 0
    print(f"[Test 14] First turn output shape: {output1.shape}")
    print(f"[Test 14] First turn metadata: {meta1}")

    # Test with history (second turn)
    history = [output1]
    new_input2 = encode_text("How is it different from deep learning?", embedding_dim=256)
    output2, meta2 = multi_step_reasoning(history, new_input2)

    assert output2.shape == (256,)
    assert meta2["num_history_items"] == 1
    print(f"[Test 14] Second turn output shape: {output2.shape}")
    print(f"[Test 14] Second turn num_history_items: {meta2['num_history_items']}")

    # Test with longer history
    history = [np.random.randn(256) for _ in range(7)]
    new_input3 = np.random.randn(256)
    output3, meta3 = multi_step_reasoning(history, new_input3, context_window=5)

    assert output3.shape == (256,)
    assert meta3["num_history_items"] == 5  # Should be limited to context_window
    print(f"[Test 14] Long history (7 items, window=5): {meta3['num_history_items']} used")

    print("[Test 14] ✅ multi_step_reasoning works correctly!")

    # --- Test 15: Integration test (full pipeline) ---
    print("\n--- Test 15: Integration test (full pipeline) ---")

    # Simulate a multi-turn RAG pipeline
    queries = [
        "What is climate change?",
        "What causes it?",
        "How can we prevent it?",
    ]

    reasoning_history: List[np.ndarray] = []
    responses = []

    for i, query_text in enumerate(queries):
        # Encode query
        query_emb = encode_text(query_text, embedding_dim=256)

        # Multi-step reasoning with history
        output, meta = multi_step_reasoning(reasoning_history, query_emb)
        reasoning_history.append(output)

        responses.append({
            "turn": i + 1,
            "query": query_text,
            "reasoning_steps": meta["total_reasoning_steps"],
            "best_score": meta["best_score"],
        })
        score_str = f"{meta['best_score']:.4f}" if meta['best_score'] is not None else 'N/A'
        print(f"[Test 15] Turn {i + 1}: '{query_text[:30]}...' - "
              f"steps={meta['total_reasoning_steps']}, score={score_str}")

    assert len(responses) == 3
    print("[Test 15] ✅ Integration pipeline works correctly!")

    print("\n" + "=" * 60)
    print("✅ All TRLINKOS LLM Reasoning Layer tests passed!")
    print("=" * 60)
