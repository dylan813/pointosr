import numpy as np
import os
import sys
import argparse
from pathlib import Path
import pickle
import torch
import torch.nn.functional as F

def build_prototypes(features, labels, normalize=True):
    """Build prototypes for each class.
    
    Args:
        features: Features array of shape (n_samples, feature_dim)
        labels: Labels array of shape (n_samples,)
        normalize: Whether to normalize the prototypes
        
    Returns:
        dict: Mapping from class labels to prototype vectors
    """
    unique_labels = np.unique(labels)
    prototypes = {}
    
    for label in unique_labels:
        # Get features for this class
        class_features = features[labels == label]
        
        # Compute mean prototype
        prototype = np.mean(class_features, axis=0)
        
        # Normalize if requested
        if normalize:
            prototype = prototype / np.linalg.norm(prototype)
        
        # Store prototype
        prototypes[int(label)] = prototype
    
    return prototypes

def parse_args():
    parser = argparse.ArgumentParser('Build prototypes for open-set recognition')
    parser.add_argument('--features_path', type=str, required=True, help='path to features pickle file')
    parser.add_argument('--save_path', type=str, default='data/prototypes.pkl', help='path to save prototypes')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load features
    with open(args.features_path, 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']
    labels = data['labels']
    
    print(f"Building prototypes from {len(features)} samples with {len(np.unique(labels))} classes")
    
    # Build prototypes
    prototypes = build_prototypes(features, labels)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Save prototypes
    with open(args.save_path, 'wb') as f:
        pickle.dump(prototypes, f)
    
    print(f"Saved prototypes for {len(prototypes)} classes to {args.save_path}")
    
    # Print some statistics
    print("\nPrototype statistics:")
    print(f"Number of classes: {len(prototypes)}")
    
    # Get prototype dimensionality
    proto_dim = next(iter(prototypes.values())).shape[0]
    print(f"Prototype dimensionality: {proto_dim}")
    
    # Compute pairwise similarities between prototypes
    proto_vectors = np.stack(list(prototypes.values()))
    similarities = proto_vectors @ proto_vectors.T
    
    # Remove self-similarities from the diagonal
    mask = ~np.eye(len(proto_vectors), dtype=bool)
    similarities = similarities[mask].reshape(len(proto_vectors), -1)
    
    print(f"Mean inter-class similarity: {np.mean(similarities):.4f}")
    print(f"Min inter-class similarity: {np.min(similarities):.4f}")
    print(f"Max inter-class similarity: {np.max(similarities):.4f}")

if __name__ == "__main__":
    main() 