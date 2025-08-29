"""
Custom model class for LLM2Vec4CXR that properly handles latent attention pooling.
"""

from llm2vec.models.bidirectional_llama import LlamaBiModel
from llm2vec.pooling import LatentAttentionPooling
import torch
import torch.nn as nn


class LLM2Vec4CXRModel(LlamaBiModel):
    """
    Custom LlamaBiModel that includes latent attention pooling by default.
    This prevents the warning about unused latent attention weights.
    """
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        
        # Initialize latent attention pooling
        self.latent_attn = LatentAttentionPooling(
            d_model=config.hidden_size,
            num_heads=8,  # Standard for this model size
            num_latents=512  # Standard for LLM2Vec
        )
        
        # Move to the same device/dtype as the base model
        if hasattr(self, 'model') and hasattr(self.model, 'embed_tokens'):
            device = self.model.embed_tokens.weight.device
            dtype = self.model.embed_tokens.weight.dtype
            self.latent_attn = self.latent_attn.to(device=device, dtype=dtype)
    
    def forward(self, input_ids, attention_mask=None, embed_mask=None, **kwargs):
        """
        Forward pass that properly handles latent attention pooling.
        """
        # Get base model output
        outputs = super().forward(input_ids, attention_mask=attention_mask, **kwargs)
        
        # If we have latent attention pooling, apply it
        if hasattr(self, 'latent_attn') and self.latent_attn is not None:
            if embed_mask is not None:
                # Use embed_mask for instruction-following tasks
                pooled_output = self.latent_attn(outputs.last_hidden_state, embed_mask)
            else:
                # Use attention_mask for simple encoding
                pooled_output = self.latent_attn(outputs.last_hidden_state, attention_mask)
            return pooled_output
        
        return outputs.last_hidden_state


# Register the model for auto loading
from transformers import AutoModel
AutoModel.register(LLM2Vec4CXRModel.__name__, LLM2Vec4CXRModel)
