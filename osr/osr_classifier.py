import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Add the parent directory to the path to import from point_osr
sys.path.append(str(Path(__file__).parent.parent))

from dataset.build import build_dataloader_from_cfg
from models.pointnext_wrapper import PointNeXtFeatureExtractor

class OSRClassifier:
    """Prototype-based Open Set Recognition classifier.
    
    This classifier uses cosine similarity to compare a feature to class prototypes.
    Samples are rejected as "unknown" if the similarity to all known classes is below a threshold.
    """
    def __init__(self, prototypes, threshold=0.75, device=None):
        """Initialize the OSR classifier.
        
        Args:
            prototypes: Dict mapping class labels to prototype vectors
            threshold: Threshold for rejecting samples as "unknown"
            device: Device to run the classifier on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert prototypes to tensors and move to device
        self.prototypes = {}
        for label, proto in prototypes.items():
            self.prototypes[label] = torch.tensor(proto, dtype=torch.float32).to(self.device)
        
        self.threshold = threshold
        self.unknown_label = -1  # Label for unknown class
    
    def classify(self, feature):
        """Classify a single feature vector.
        
        Args:
            feature: Feature vector to classify
            
        Returns:
            tuple: (predicted_class, similarity)
        """
        # Convert to tensor if necessary
        if isinstance(feature, np.ndarray):
            feature = torch.tensor(feature, dtype=torch.float32).to(self.device)
        
        # Ensure feature is normalized
        feature = F.normalize(feature, p=2, dim=0)
        
        # Compute similarities to all prototypes
        similarities = {}
        for label, proto in self.prototypes.items():
            similarities[label] = torch.dot(feature, proto).item()
        
        # Find the class with the highest similarity
        best_class = max(similarities, key=similarities.get)
        best_sim = similarities[best_class]
        
        # Return unknown if similarity is below threshold
        if best_sim < self.threshold:
            return self.unknown_label, best_sim
        else:
            return best_class, best_sim
    
    def batch_classify(self, features):
        """Classify a batch of feature vectors.
        
        Args:
            features: Batch of feature vectors (n_samples, feature_dim)
            
        Returns:
            tuple: (predicted_classes, best_similarities)
        """
        # Convert to tensor if necessary
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # Ensure features are normalized
        features = F.normalize(features, p=2, dim=1)
        
        # Stack prototype vectors
        proto_labels = list(self.prototypes.keys())
        proto_tensors = torch.stack([self.prototypes[label] for label in proto_labels])
        
        # Compute similarities to all prototypes at once: (n_samples, n_prototypes)
        similarities = torch.matmul(features, proto_tensors.T)
        
        # Find the prototype with the highest similarity for each sample
        best_similarities, best_indices = torch.max(similarities, dim=1)
        
        # Convert indices to class labels
        predicted_classes = torch.tensor([proto_labels[i] for i in best_indices], 
                                         device=self.device)
        
        # Mark samples with similarity below threshold as unknown
        unknown_mask = best_similarities < self.threshold
        predicted_classes[unknown_mask] = self.unknown_label
        
        return predicted_classes.cpu().numpy(), best_similarities.cpu().numpy()

def evaluate_osr(classifier, features, labels, known_classes, plot=False, save_path=None):
    """Evaluate the OSR classifier.
    
    Args:
        classifier: OSR classifier
        features: Test features
        labels: Test labels
        known_classes: List of known class labels
        plot: Whether to generate plots
        save_path: Path to save plots
        
    Returns:
        dict: Evaluation metrics
    """
    # Convert to numpy if necessary
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Create binary labels (known=1, unknown=0)
    binary_labels = np.isin(labels, known_classes).astype(int)
    
    # Get predictions and similarities
    pred_labels, similarities = classifier.batch_classify(features)
    
    # Compute closed-set accuracy on known samples
    known_mask = binary_labels == 1
    if np.any(known_mask):
        known_acc = np.mean(pred_labels[known_mask] == labels[known_mask])
    else:
        known_acc = 0.0
    
    # Compute AUROC for unknown detection
    # Higher similarity means more likely to be known, so we negate for ROC
    auroc = roc_auc_score(binary_labels, similarities)
    
    # Compute FPR at 95% TPR
    fpr, tpr, thresholds = roc_curve(binary_labels, similarities)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx]
    
    # Compute AUPR
    aupr = average_precision_score(binary_labels, similarities)
    
    metrics = {
        'known_accuracy': known_acc,
        'auroc': auroc,
        'fpr95': fpr95,
        'aupr': aupr,
    }
    
    # Generate plots if requested
    if plot:
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Plot similarity distributions
        plt.figure(figsize=(10, 6))
        plt.hist(similarities[binary_labels == 1], bins=50, alpha=0.5, label='Known')
        plt.hist(similarities[binary_labels == 0], bins=50, alpha=0.5, label='Unknown')
        plt.axvline(x=classifier.threshold, color='r', linestyle='--', 
                   label=f'Threshold = {classifier.threshold:.2f}')
        plt.xlabel('Similarity to Nearest Prototype')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Similarity Distribution for Known and Unknown Samples')
        
        if save_path:
            plt.savefig(f"{save_path}_similarity_dist.png")
            plt.close()
        else:
            plt.show()
        
        # Plot ROC curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}')
        plt.axhline(y=0.95, color='r', linestyle='--', 
                   label=f'FPR@95%TPR = {fpr95:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Unknown Detection')
        plt.legend()
        
        if save_path:
            plt.savefig(f"{save_path}_roc.png")
            plt.close()
        else:
            plt.show()
    
    return metrics

def parse_args():
    parser = argparse.ArgumentParser('Evaluate OSR classifier')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--prototypes', type=str, required=True, help='path to prototypes file')
    parser.add_argument('--threshold', type=float, default=0.75, help='similarity threshold')
    parser.add_argument('--known_classes', type=int, nargs='+', help='known class indices (if not specified, all classes in prototypes are considered known)')
    parser.add_argument('--test_features', type=str, help='path to test features (if not using model)')
    parser.add_argument('--pretrained', type=str, help='pretrained model path (if not using test_features)')
    parser.add_argument('--plot', action='store_true', help='generate plots')
    parser.add_argument('--output_dir', type=str, default='results', help='output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load prototypes
    with open(args.prototypes, 'rb') as f:
        prototypes = pickle.load(f)
    
    print(f"Loaded prototypes for {len(prototypes)} classes")
    
    # Set known classes
    known_classes = args.known_classes or list(prototypes.keys())
    print(f"Known classes: {known_classes}")
    
    # Create classifier
    classifier = OSRClassifier(prototypes, threshold=args.threshold)
    
    # Get test features
    if args.test_features:
        # Load pre-extracted features
        with open(args.test_features, 'rb') as f:
            data = pickle.load(f)
        test_features = data['features']
        test_labels = data['labels']
    else:
        # Extract features using model
        assert args.pretrained, "Either test_features or pretrained must be specified"
        
        # Import config
        sys.path.append(os.path.join(os.path.dirname(args.cfg), '..'))
        config = __import__(os.path.basename(args.cfg).split('.')[0], fromlist=[''])
        
        # Build model
        model = PointNeXtFeatureExtractor(model_cfg=config.model, pretrained_path=args.pretrained)
        model.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(model.device)
        
        # Build dataloader
        test_loader = build_dataloader_from_cfg(config.data.test, config.data.test_batch_size, 
                                             config.data.num_workers, config.data.transform)
        
        # Extract features
        features_list = []
        labels_list = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Extracting test features"):
                # Move batch to the same device as model
                if isinstance(batch, dict):
                    data, labels = batch, batch['y'].to(model.device)
                else:
                    data, labels = batch
                    if isinstance(labels, dict):
                        data, labels = batch[0], batch[1]['y']
                    labels = labels.to(model.device)
                
                # Extract features
                features = model.extract_features(data)
                
                # Store features and labels
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
        
        test_features = np.vstack(features_list)
        test_labels = np.concatenate(labels_list)
    
    print(f"Evaluating on {len(test_features)} test samples")
    
    # Evaluate
    metrics = evaluate_osr(classifier, test_features, test_labels, known_classes, 
                          plot=args.plot, save_path=os.path.join(args.output_dir, 'osr_eval'))
    
    # Print metrics
    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main() 