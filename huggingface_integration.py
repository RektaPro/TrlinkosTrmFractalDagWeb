"""
Enhanced HuggingFace Integration for T-RLINKOS TRM++

This module provides native integration with HuggingFace models and pre-trained encoders:
- Support for popular transformer models (BERT, GPT, LLaMA, Mistral, etc.)
- Pre-trained encoder wrappers (vision, text, multimodal)
- Automatic model configuration detection
- Efficient batching and tokenization
- Model security with revision pinning

Usage:
    # Text encoding with BERT
    encoder = PretrainedTextEncoder("bert-base-uncased")
    embeddings = encoder.encode(["Hello world", "AI reasoning"])
    
    # Vision encoding with ViT
    encoder = PretrainedVisionEncoder("google/vit-base-patch16-224")
    embeddings = encoder.encode(images)
    
    # Full model integration
    model = create_trlinkos_with_encoder(
        encoder_name="bert-base-uncased",
        encoder_type="text",
        output_dim=32
    )
"""

import warnings
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np

# Try to import transformers
try:
    import transformers
    from transformers import (
        AutoModel,
        AutoTokenizer,
        AutoImageProcessor,
        AutoConfig,
    )
    TRANSFORMERS_AVAILABLE = True
    TRANSFORMERS_VERSION = transformers.__version__
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TRANSFORMERS_VERSION = None
    warnings.warn(
        "transformers not available. Install with: pip install transformers>=4.30.0"
    )

# Try to import PIL for image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    # Create dummy class for type hints
    class Image:
        class Image:
            pass


# ============================
#  Model Registry
# ============================

# Common model configurations
MODEL_REGISTRY = {
    # Text models
    "bert-base": {
        "name": "bert-base-uncased",
        "hidden_dim": 768,
        "type": "text",
        "description": "BERT base uncased",
    },
    "bert-large": {
        "name": "bert-large-uncased",
        "hidden_dim": 1024,
        "type": "text",
        "description": "BERT large uncased",
    },
    "gpt2": {
        "name": "gpt2",
        "hidden_dim": 768,
        "type": "text",
        "description": "GPT-2 small",
    },
    "gpt2-medium": {
        "name": "gpt2-medium",
        "hidden_dim": 1024,
        "type": "text",
        "description": "GPT-2 medium",
    },
    "distilbert": {
        "name": "distilbert-base-uncased",
        "hidden_dim": 768,
        "type": "text",
        "description": "DistilBERT base",
    },
    "roberta-base": {
        "name": "roberta-base",
        "hidden_dim": 768,
        "type": "text",
        "description": "RoBERTa base",
    },
    
    # Vision models
    "vit-base": {
        "name": "google/vit-base-patch16-224",
        "hidden_dim": 768,
        "type": "vision",
        "description": "Vision Transformer base",
    },
    "vit-large": {
        "name": "google/vit-large-patch16-224",
        "hidden_dim": 1024,
        "type": "vision",
        "description": "Vision Transformer large",
    },
    
    # Large language models (require more resources)
    "llama-7b": {
        "name": "meta-llama/Llama-2-7b-hf",
        "hidden_dim": 4096,
        "type": "text",
        "description": "LLaMA 2 7B (requires auth)",
        "requires_auth": True,
    },
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-v0.1",
        "hidden_dim": 4096,
        "type": "text",
        "description": "Mistral 7B",
    },
}


def get_model_info(model_name_or_alias: str) -> Dict[str, Any]:
    """Get information about a model from the registry.
    
    Args:
        model_name_or_alias: Model name or alias (e.g., "bert-base" or "bert-base-uncased")
        
    Returns:
        Dictionary with model information
    """
    # Check if it's an alias
    if model_name_or_alias in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name_or_alias]
    
    # Check if it matches a full name
    for alias, info in MODEL_REGISTRY.items():
        if info["name"] == model_name_or_alias:
            return info
    
    # Return placeholder info
    return {
        "name": model_name_or_alias,
        "hidden_dim": None,
        "type": "unknown",
        "description": f"Custom model: {model_name_or_alias}",
    }


def list_available_models(model_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all available pre-configured models.
    
    Args:
        model_type: Filter by type ("text", "vision", or None for all)
        
    Returns:
        List of model information dictionaries
    """
    models = []
    for alias, info in MODEL_REGISTRY.items():
        if model_type is None or info["type"] == model_type:
            models.append({"alias": alias, **info})
    return models


# ============================
#  Pre-trained Text Encoder
# ============================

class PretrainedTextEncoder:
    """Pre-trained text encoder using HuggingFace transformers.
    
    Wraps any HuggingFace text model (BERT, GPT, RoBERTa, etc.) for use
    with T-RLINKOS. Handles tokenization, encoding, and pooling automatically.
    
    Example:
        encoder = PretrainedTextEncoder("bert-base-uncased")
        embeddings = encoder.encode(["Hello world", "AI is amazing"])
        # embeddings shape: (2, 768)
    """
    
    def __init__(
        self,
        model_name: str,
        output_dim: Optional[int] = None,
        pooling: str = "mean",
        max_length: int = 512,
        device: str = "cpu",
        revision: Optional[str] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
    ):
        """Initialize pre-trained text encoder.
        
        Args:
            model_name: HuggingFace model name or alias from MODEL_REGISTRY
            output_dim: Optional output dimension (adds projection layer)
            pooling: Pooling strategy ("mean", "max", "cls", "last")
            max_length: Maximum sequence length
            device: Device to run model on ("cpu" or "cuda")
            revision: Specific model revision/commit hash for security
            use_auth_token: HuggingFace auth token for private models
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not available. Install with: pip install transformers>=4.30.0")
        
        # Get model info
        model_info = get_model_info(model_name)
        self.model_name = model_info["name"]
        self.hidden_dim = model_info.get("hidden_dim")
        self.output_dim = output_dim or self.hidden_dim
        self.pooling = pooling
        self.max_length = max_length
        self.device = device
        
        print(f"Loading text encoder: {self.model_name}")
        if revision:
            print(f"  Using revision: {revision}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                revision=revision,
                use_auth_token=use_auth_token,
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                revision=revision,
                use_auth_token=use_auth_token,
            )
            
            # Get actual hidden dimension from model
            config = AutoConfig.from_pretrained(self.model_name)
            self.hidden_dim = config.hidden_size
            
            if output_dim and output_dim != self.hidden_dim:
                print(f"  Adding projection: {self.hidden_dim} -> {output_dim}")
                # Create simple projection (NumPy-based for compatibility)
                from t_rlinkos_trm_fractal_dag import LinearNP
                self.projection = LinearNP(self.hidden_dim, output_dim)
            else:
                self.projection = None
            
            print(f"  Model loaded successfully (hidden_dim={self.hidden_dim})")
            
        except Exception as e:
            print(f"  Error loading model: {e}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = False,
    ) -> np.ndarray:
        """Encode texts to embeddings.
        
        Args:
            texts: Single text or list of texts
            normalize: Normalize embeddings to unit length
            
        Returns:
            Embeddings array [B, output_dim]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Forward pass (no gradients needed)
        import torch
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract embeddings based on pooling strategy
        if self.pooling == "cls":
            # Use [CLS] token (first token)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        elif self.pooling == "mean":
            # Mean pooling over sequence
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            masked_embeddings = outputs.last_hidden_state * attention_mask
            embeddings = masked_embeddings.sum(1) / attention_mask.sum(1)
            embeddings = embeddings.cpu().numpy()
        elif self.pooling == "max":
            # Max pooling over sequence
            embeddings = torch.max(outputs.last_hidden_state, dim=1)[0].cpu().numpy()
        elif self.pooling == "last":
            # Use last token
            embeddings = outputs.last_hidden_state[:, -1, :].cpu().numpy()
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Apply projection if needed
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        return embeddings


# ============================
#  Pre-trained Vision Encoder
# ============================

class PretrainedVisionEncoder:
    """Pre-trained vision encoder using HuggingFace transformers.
    
    Wraps vision models (ViT, CLIP, etc.) for use with T-RLINKOS.
    Handles image preprocessing and encoding automatically.
    
    Example:
        encoder = PretrainedVisionEncoder("google/vit-base-patch16-224")
        images = [Image.open("cat.jpg"), Image.open("dog.jpg")]
        embeddings = encoder.encode(images)
    """
    
    def __init__(
        self,
        model_name: str,
        output_dim: Optional[int] = None,
        device: str = "cpu",
        revision: Optional[str] = None,
    ):
        """Initialize pre-trained vision encoder.
        
        Args:
            model_name: HuggingFace model name or alias
            output_dim: Optional output dimension (adds projection)
            device: Device to run model on
            revision: Specific model revision/commit hash for security
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not available")
        
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL not available. Install with: pip install Pillow")
        
        model_info = get_model_info(model_name)
        self.model_name = model_info["name"]
        self.hidden_dim = model_info.get("hidden_dim")
        self.output_dim = output_dim or self.hidden_dim
        self.device = device
        
        print(f"Loading vision encoder: {self.model_name}")
        
        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            revision=revision,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            revision=revision,
        )
        
        # Get actual hidden dimension
        config = AutoConfig.from_pretrained(self.model_name)
        self.hidden_dim = config.hidden_size
        
        if output_dim and output_dim != self.hidden_dim:
            from t_rlinkos_trm_fractal_dag import LinearNP
            self.projection = LinearNP(self.hidden_dim, output_dim)
        else:
            self.projection = None
        
        print(f"  Model loaded (hidden_dim={self.hidden_dim})")
    
    def encode(
        self,
        images: Union[Image.Image, List[Image.Image]],
        normalize: bool = False,
    ) -> np.ndarray:
        """Encode images to embeddings.
        
        Args:
            images: Single PIL Image or list of PIL Images
            normalize: Normalize embeddings to unit length
            
        Returns:
            Embeddings array [B, output_dim]
        """
        if isinstance(images, Image.Image):
            images = [images]
        
        # Preprocess images
        inputs = self.processor(images, return_tensors="pt")
        
        # Forward pass
        import torch
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract embeddings (use pooler output or mean pooling)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output.cpu().numpy()
        else:
            # Mean pooling over patches
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # Apply projection if needed
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        return embeddings


# ============================
#  Integration Utilities
# ============================

def create_trlinkos_with_encoder(
    encoder_name: str,
    encoder_type: str = "text",
    output_dim: int = 32,
    z_dim: int = 64,
    hidden_dim: int = 256,
    num_experts: int = 4,
    **encoder_kwargs,
):
    """Create a T-RLINKOS model with a pre-trained encoder.
    
    Args:
        encoder_name: HuggingFace model name or alias
        encoder_type: "text" or "vision"
        output_dim: Output dimension of T-RLINKOS
        z_dim: Internal state dimension
        hidden_dim: Hidden dimension
        num_experts: Number of experts
        **encoder_kwargs: Additional arguments for encoder
        
    Returns:
        Tuple of (encoder, trlinkos_model)
    """
    from t_rlinkos_trm_fractal_dag import TRLinkosTRM
    
    # Create encoder
    if encoder_type == "text":
        encoder = PretrainedTextEncoder(encoder_name, **encoder_kwargs)
    elif encoder_type == "vision":
        encoder = PretrainedVisionEncoder(encoder_name, **encoder_kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    # Create T-RLINKOS model
    x_dim = encoder.output_dim
    model = TRLinkosTRM(
        x_dim=x_dim,
        y_dim=output_dim,
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
    )
    
    print(f"\nCreated T-RLINKOS model:")
    print(f"  Input (from encoder): {x_dim}")
    print(f"  Output: {output_dim}")
    print(f"  Internal state: {z_dim}")
    print(f"  Hidden: {hidden_dim}")
    print(f"  Experts: {num_experts}")
    
    return encoder, model


# ============================
#  Main test
# ============================

if __name__ == "__main__":
    print("=" * 70)
    print("HUGGINGFACE INTEGRATION MODULE TEST")
    print("=" * 70)
    
    print(f"\nTransformers available: {TRANSFORMERS_AVAILABLE}")
    if TRANSFORMERS_AVAILABLE:
        print(f"Transformers version: {TRANSFORMERS_VERSION}")
    
    # List available models
    print("\n--- Available Text Models ---")
    text_models = list_available_models(model_type="text")
    for model in text_models[:5]:  # Show first 5
        print(f"  {model['alias']:15s} - {model['description']}")
    
    print("\n--- Available Vision Models ---")
    vision_models = list_available_models(model_type="vision")
    for model in vision_models:
        print(f"  {model['alias']:15s} - {model['description']}")
    
    # Test model info lookup
    print("\n--- Test: Model Info Lookup ---")
    info = get_model_info("bert-base")
    print(f"Model: {info['name']}")
    print(f"Hidden dim: {info['hidden_dim']}")
    print(f"Type: {info['type']}")
    
    print("\n" + "=" * 70)
    print("✅ HuggingFace integration module loaded successfully!")
    print("=" * 70)
    
    if not TRANSFORMERS_AVAILABLE:
        print("\nNote: Install transformers to use pre-trained encoders:")
        print("  pip install transformers>=4.30.0")
    
    print("\nUsage Examples:")
    print("\n1. Text encoding with BERT:")
    print("   encoder = PretrainedTextEncoder('bert-base')")
    print("   embeddings = encoder.encode(['Hello world', 'AI reasoning'])")
    
    print("\n2. Vision encoding with ViT:")
    print("   encoder = PretrainedVisionEncoder('vit-base')")
    print("   embeddings = encoder.encode(images)")
    
    print("\n3. Full integration:")
    print("   encoder, model = create_trlinkos_with_encoder(")
    print("       encoder_name='bert-base',")
    print("       encoder_type='text',")
    print("       output_dim=32")
    print("   )")
