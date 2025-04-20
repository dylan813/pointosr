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

# Import only the classifier class
from osr.osr_classifier import OSRClassifier

def plot_feature_visualization(features, labels, prototypes=None, save_path=None):
    """Create a t-SNE visualization of the feature space, coloring by label.
    
    Args:
        features: Feature vectors
        labels: Class labels (e.g., 0: Human, 1: False, -1: Unknown/Rejected, others: Other Unknown)
        prototypes: Optional dict of class prototypes {0: proto_h, 1: proto_f}
        save_path: Path to save the plot
    """
    # Convert to numpy if needed
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Subsample if too many features
    max_samples = 2000 # Keep subsampling for performance
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    # Apply t-SNE
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1)) # Adjust perplexity if needed
    features_2d = tsne.fit_transform(features)
    
    # Set up plotting
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    
    # Use a categorical colormap
    cmap = plt.get_cmap('tab10', len(unique_labels))
    
    # Map labels to colors, handling specific labels
    label_map = {0: "Human", 1: "False", -1: "Unknown/Rejected"}
    colors = {}
    legend_handles = []
    
    # Assign colors consistently
    color_idx = 0
    for label in sorted(unique_labels):
         colors[label] = cmap(color_idx)
         display_label = label_map.get(label, f"Other Unknown ({label})")
         # Create dummy scatter for legend
         legend_handles.append(plt.scatter([], [], color=colors[label], label=display_label))
         color_idx += 1

    # Plot points with assigned colors
    for label in unique_labels:
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                     color=colors[label], alpha=0.6)

    # Plot prototypes if provided (should contain 0 and 1)
    if prototypes is not None and 0 in prototypes and 1 in prototypes:
        proto_features = np.array([prototypes[0], prototypes[1]])
        proto_labels_text = ["Proto Human (0)", "Proto False (1)"]
        
        # Avoid re-fitting t-SNE if possible, but direct transform is tricky.
        # Re-fitting with combined data is a common workaround, though not ideal.
        print("Applying t-SNE to prototypes...")
        combined_features = np.vstack([features, proto_features])
        combined_features_2d = tsne.fit_transform(combined_features)
        proto_features_2d = combined_features_2d[-len(proto_features):]
        
        plt.scatter(proto_features_2d[:, 0], proto_features_2d[:, 1], 
                  marker='*', s=300, c='black', label='Prototypes')
        
        # Add text labels for prototypes
        for i, text in enumerate(proto_labels_text):
            plt.annotate(text, (proto_features_2d[i, 0], proto_features_2d[i, 1]),
                       xytext=(5, 5), textcoords='offset points', fontweight='bold',
                       bbox=dict(boxstyle="round", fc='yellow', alpha=0.8))
    
    plt.title('t-SNE Visualization of Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    # Use handles for legend
    plt.legend(handles=legend_handles, title="True Labels")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved t-SNE plot to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names_map, save_path=None):
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names_map: Dict mapping label values (e.g., 0, 1, -1) to names ("Human", "False", "Rejected")
        save_path: Path to save the plot
    """
    # Determine the union of unique labels present in true and predicted
    present_labels = sorted(list(set(y_true) | set(y_pred)))
    display_labels = [class_names_map.get(lbl, str(lbl)) for lbl in present_labels]
    
    # Compute confusion matrix using the ordered list of present labels
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    
    # Plot
    plt.figure(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
        plt.close()
    else:
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser('Generate Visualizations for Dual-Threshold OSR')
    # Removed --train_features
    parser.add_argument('--test_features', type=str, required=True, 
                        help='Path to pre-extracted test/validation features (.pkl file)')
    parser.add_argument('--prototypes', type=str, required=True, 
                        help='Path to prototypes file (.pkl file, must contain 0 and 1)')
    # Removed --known_classes and --threshold
    parser.add_argument('--threshold_human', type=float, required=True, 
                        help='Similarity threshold for human class (0)')
    parser.add_argument('--threshold_false', type=float, required=True, 
                        help='Similarity threshold for false class (1)')
    parser.add_argument('--output_dir', type=str, default='results_visuals', 
                        help='Output directory for plots')
    parser.add_argument('--no_plots', action='store_true', help='Disable plot generation')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load test features
    print(f"Loading test features from: {args.test_features}")
    with open(args.test_features, 'rb') as f:
        test_data = pickle.load(f)
    test_features = test_data['features']
    test_labels = test_data['labels']
    print(f"Loaded {len(test_features)} test samples.")
    
    # Load prototypes
    print(f"Loading prototypes from: {args.prototypes}")
    with open(args.prototypes, 'rb') as f:
        prototypes = pickle.load(f)
    if 0 not in prototypes or 1 not in prototypes:
         raise ValueError(f"Prototypes file {args.prototypes} must contain keys for 0 (human) and 1 (false)")
    print(f"Loaded prototypes for classes: {list(prototypes.keys())}")

    # Create threshold dictionary
    thresholds_dict = {
        0: args.threshold_human,
        1: args.threshold_false
    }
    print(f"Using thresholds: Human(0)={thresholds_dict[0]:.4f}, False(1)={thresholds_dict[1]:.4f}")
    
    # Create classifier (ensure device is handled if needed, but classifier uses it internally)
    classifier = OSRClassifier(prototypes, thresholds_dict=thresholds_dict)
    
    # Create directories for outputs
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Generate Plots (if not disabled) ---
    if not args.no_plots:
        print("\nGenerating plots...")
        
        # 1. t-SNE Visualization
        tsne_save_path = os.path.join(args.output_dir, 'tsne_features.png')
        try:
            plot_feature_visualization(test_features, test_labels, prototypes,
                                    save_path=tsne_save_path)
        except Exception as e:
            print(f"Error generating t-SNE plot: {e}")
            # Optionally continue or raise error

        # 2. Confusion Matrix
        print("\nCalculating predictions for confusion matrix...")
        # Get predictions using the dual-threshold classifier
        pred_labels, _ = classifier.batch_classify(test_features)
        
        # Define class names for plotting
        # Include all labels potentially present in true or predicted sets
        class_names_map = {
            0: "Human", 
            1: "False", 
            classifier.unknown_label: "Rejected" 
        }
        # Add mappings for any other true unknown labels if they exist
        other_unknowns = set(test_labels) - set(class_names_map.keys())
        for lbl in other_unknowns:
            class_names_map[lbl] = f"Unknown ({lbl})"

        cm_save_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        try:
            plot_confusion_matrix(test_labels, pred_labels, 
                                class_names_map=class_names_map,
                                save_path=cm_save_path)
        except Exception as e:
            print(f"Error generating confusion matrix plot: {e}")
    else:
        print("Plot generation disabled by --no_plots argument.")

    # Removed the single-threshold analysis loop and call to old evaluate_osr
    print("\nEvaluation script finished.")

if __name__ == "__main__":
    main() 