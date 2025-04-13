import torch
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Add the parent directory to the path to import from point_osr
sys.path.append(str(Path(__file__).parent.parent))

from osr.osr_classifier import OSRClassifier, evaluate_osr

def plot_feature_visualization(features, labels, prototypes=None, known_classes=None, save_path=None):
    """Create a t-SNE visualization of the feature space.
    
    Args:
        features: Feature vectors
        labels: Class labels
        prototypes: Optional dict of class prototypes
        known_classes: List of known class labels (to separate known/unknown)
        save_path: Path to save the plot
    """
    # Convert to numpy if needed
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Subsample if too many features
    max_samples = 2000
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    # Apply t-SNE
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Set up plotting
    plt.figure(figsize=(12, 10))
    
    # Set binary known/unknown if known_classes is provided
    if known_classes is not None:
        binary_labels = np.isin(labels, known_classes).astype(int)
        # Use a discrete colormap for binary case
        cmap = ['red', 'blue']
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=binary_labels, 
                             cmap=plt.cm.get_cmap('coolwarm', 2), alpha=0.6)
        plt.colorbar(scatter, ticks=[0, 1], label='Class')
        plt.clim(-0.5, 1.5)
        
        # Add text labels
        for i, label in enumerate(['Unknown', 'Known']):
            # Find a representative point
            if np.any(binary_labels == i):
                idx = np.where(binary_labels == i)[0][0]
                plt.annotate(label, (features_2d[idx, 0], features_2d[idx, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           bbox=dict(boxstyle="round", fc=cmap[i], alpha=0.6))
    else:
        # Use a continuous colormap for multi-class
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab20', alpha=0.6)
        plt.colorbar(scatter, label='Class')
    
    # Plot prototypes if provided
    if prototypes is not None:
        # Project prototypes to the same t-SNE space
        proto_features = np.array(list(prototypes.values()))
        proto_labels = np.array(list(prototypes.keys()))
        
        # We need to project prototypes
        # This requires using the pre-computed t-SNE model with transform
        # which isn't directly available in scikit-learn t-SNE
        # As a workaround, we'll just plot them with a special marker
        proto_features_2d = tsne.fit_transform(np.vstack([features, proto_features]))[-len(proto_features):]
        
        plt.scatter(proto_features_2d[:, 0], proto_features_2d[:, 1], 
                  marker='*', s=200, c='black', label='Prototypes')
        
        # Add text labels for prototypes
        for i, label in enumerate(proto_labels):
            plt.annotate(f'Proto {label}', (proto_features_2d[i, 0], proto_features_2d[i, 1]),
                       xytext=(5, 5), textcoords='offset points', fontweight='bold',
                       bbox=dict(boxstyle="round", fc='yellow', alpha=0.6))
    
    plt.title('t-SNE Visualization of Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser('Comprehensive OSR Evaluation')
    parser.add_argument('--train_features', type=str, required=True, help='path to training features')
    parser.add_argument('--test_features', type=str, required=True, help='path to test features')
    parser.add_argument('--prototypes', type=str, required=True, help='path to prototypes')
    parser.add_argument('--known_classes', type=int, nargs='+', required=True, help='known class indices')
    parser.add_argument('--threshold', type=float, default=0.75, help='similarity threshold')
    parser.add_argument('--output_dir', type=str, default='results', help='output directory')
    parser.add_argument('--no_plots', action='store_true', help='disable plots')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load features
    with open(args.train_features, 'rb') as f:
        train_data = pickle.load(f)
    with open(args.test_features, 'rb') as f:
        test_data = pickle.load(f)
    
    train_features = train_data['features']
    train_labels = train_data['labels']
    test_features = test_data['features']
    test_labels = test_data['labels']
    
    # Load prototypes
    with open(args.prototypes, 'rb') as f:
        prototypes = pickle.load(f)
    
    # Create classifier
    classifier = OSRClassifier(prototypes, threshold=args.threshold)
    
    # Determine known and unknown classes
    known_classes = args.known_classes
    all_classes = sorted(list(set(test_labels.tolist())))
    unknown_classes = [c for c in all_classes if c not in known_classes]
    
    print(f"Evaluating with {len(known_classes)} known classes and {len(unknown_classes)} unknown classes")
    print(f"Known classes: {known_classes}")
    print(f"Unknown classes: {unknown_classes}")
    
    # Create directories for outputs
    results_dir = os.path.join(args.output_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # Perform evaluation
    metrics = evaluate_osr(classifier, test_features, test_labels, known_classes, 
                          plot=(not args.no_plots), save_path=os.path.join(results_dir, 'eval'))
    
    # Print metrics
    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Save metrics
    with open(os.path.join(results_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    
    # Generate additional plots if enabled
    if not args.no_plots:
        # Get predictions for confusion matrix
        pred_labels, _ = classifier.batch_classify(test_features)
        
        # Create class names mapping
        class_names = {}
        for c in all_classes:
            if c in known_classes:
                class_names[c] = f"Known-{c}"
            else:
                class_names[c] = f"Unknown-{c}"
        class_names[classifier.unknown_label] = "Rejected" 
        
        # Plot confusion matrix
        plot_confusion_matrix(test_labels, pred_labels, 
                            class_names=[class_names.get(c, str(c)) for c in sorted(list(set(list(pred_labels) + list(test_labels))))],
                            save_path=os.path.join(results_dir, 'confusion_matrix.png'))
        
        # Visualize features
        plot_feature_visualization(test_features, test_labels, prototypes, known_classes,
                                save_path=os.path.join(results_dir, 'tsne_features.png'))
        
        # Plot threshold analysis - how does accuracy change with different thresholds
        thresholds = np.linspace(0.0, 1.0, 50)
        known_accs = []
        unknown_accs = []
        
        # Binary labels for known/unknown
        binary_labels = np.isin(test_labels, known_classes).astype(int)
        
        for t in thresholds:
            # Update classifier threshold
            classifier.threshold = t
            
            # Get predictions
            pred_labels, _ = classifier.batch_classify(test_features)
            
            # Compute accuracy for known samples
            known_mask = binary_labels == 1
            if np.any(known_mask):
                known_acc = np.mean((pred_labels[known_mask] == test_labels[known_mask]).astype(float))
                known_accs.append(known_acc)
            else:
                known_accs.append(0)
            
            # Compute accuracy for unknown samples (should be rejected)
            unknown_mask = binary_labels == 0
            if np.any(unknown_mask):
                unknown_acc = np.mean((pred_labels[unknown_mask] == classifier.unknown_label).astype(float))
                unknown_accs.append(unknown_acc)
            else:
                unknown_accs.append(0)
        
        # Plot threshold analysis
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, known_accs, 'b-', label='Known Classification Accuracy')
        plt.plot(thresholds, unknown_accs, 'r-', label='Unknown Rejection Rate')
        plt.axvline(x=args.threshold, color='k', linestyle='--', 
                    label=f'Selected Threshold = {args.threshold:.2f}')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.title('Effect of Threshold on Classification Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'threshold_analysis.png'))
        plt.close()

if __name__ == "__main__":
    main() 