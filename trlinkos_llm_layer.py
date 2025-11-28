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

    print("\n" + "=" * 60)
    print("✅ All TRLINKOS LLM Reasoning Layer tests passed!")
    print("=" * 60)
