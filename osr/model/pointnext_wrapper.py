import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add the parent directory to the path to import from point_osr
sys.path.append(str(Path(__file__).parent.parent.parent))
from model.build import build_model_from_cfg
from model.classification.cls_base import BaseCls

class PointNeXtFeatureExtractor(nn.Module):
    """Wrapper around PointNeXt model to extract penultimate layer features for OSR.
    
    This model returns both the classification output and the feature embedding.
    The feature embedding is normalized for cosine similarity calculations.
    """
    def __init__(self, model_cfg=None, pretrained_path=None):
        """Initialize the feature extractor.
        
        Args:
            model_cfg: Config dict for model architecture
            pretrained_path: Path to pretrained model weights
        """
        super().__init__()
        
        # Build the base classification model
        self.model = build_model_from_cfg(model_cfg) if model_cfg else None
        
        # Load pretrained weights if provided
        if pretrained_path and self.model:
            self._load_pretrained(pretrained_path)
            
        # Extract dimensions of the feature layer
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'out_channels'):
            self.feature_dim = self.model.encoder.out_channels
        else:
            self.feature_dim = 1024  # Default for most PointNeXt variants
    
    def _load_pretrained(self, pretrained_path):
        """Load pretrained weights."""
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        
        # Handle missing or unexpected parameters
        self.model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded pretrained model from {pretrained_path}")
    
    def forward(self, data):
        """Forward pass to get both classification output and feature embeddings.
        
        Args:
            data: Input data (point clouds)
            
        Returns:
            tuple: (classification_output, normalized_feature_embedding)
        """
        if isinstance(self.model, BaseCls):
            # Get global feature from encoder before classification head
            global_feat = self.model.encoder.forward_cls_feat(data)
            
            # Get classification output
            cls_output = self.model.prediction(global_feat)
            
            # Normalize feature for cosine similarity
            normalized_feat = F.normalize(global_feat, p=2, dim=1)
            
            return cls_output, normalized_feat
        else:
            raise ValueError("Unsupported model type. Expected BaseCls instance.")
    
    def extract_features(self, data):
        """Extract only the normalized feature embeddings.
        
        Args:
            data: Input data (point clouds)
            
        Returns:
            torch.Tensor: Normalized feature embedding
        """
        with torch.no_grad():
            _, features = self.forward(data)
        return features 