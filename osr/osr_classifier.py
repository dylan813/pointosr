import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Add the parent directory to the path to import from point_osr
sys.path.append(str(Path(__file__).parent.parent))

from dataset.build import build_dataloader_from_cfg
from model.pointnext_wrapper import PointNeXtFeatureExtractor

class OSRClassifier:
    """Prototype-based Open Set Recognition classifier using per-class thresholds.
    
    This classifier uses cosine similarity to compare a feature to class prototypes.
    Samples are classified based on per-class thresholds and relative similarity.
    Assumes specific labels for human (0) and false (1).
    """
    def __init__(self, prototypes, thresholds_dict, device=None):
        """Initialize the OSR classifier.
        
        Args:
            prototypes: Dict mapping class labels (e.g., 0, 1) to prototype vectors
            thresholds_dict: Dict mapping class labels (e.g., 0, 1) to their specific thresholds
            device: Device to run the classifier on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert prototypes to tensors and move to device
        self.prototypes = {}
        for label, proto in prototypes.items():
            # Ensure labels are integers if they represent classes like 0 and 1
            self.prototypes[int(label)] = torch.tensor(proto, dtype=torch.float32).to(self.device)
        
        # Store per-class thresholds
        self.thresholds = {int(k): v for k, v in thresholds_dict.items()}
        
        if 0 not in self.prototypes or 1 not in self.prototypes:
            raise ValueError("Prototypes must contain keys for human (0) and false (1)")
        if 0 not in self.thresholds or 1 not in self.thresholds:
            raise ValueError("Thresholds dictionary must contain keys for human (0) and false (1)")
            
        self.human_label = 0
        self.false_label = 1
        self.unknown_label = -1  # Label for unknown class
    
    def classify(self, feature):
        """Classify a single feature vector using dual thresholds.
        
        Args:
            feature: Feature vector to classify
            
        Returns:
            tuple: (predicted_class, dict_of_similarities)
        """
        # Convert to tensor if necessary
        if isinstance(feature, np.ndarray):
            feature = torch.tensor(feature, dtype=torch.float32).to(self.device)
        
        # Ensure feature is normalized
        feature = F.normalize(feature, p=2, dim=0)
        
        # Compute similarities to human and false prototypes
        sim_h = torch.dot(feature, self.prototypes[self.human_label]).item()
        sim_f = torch.dot(feature, self.prototypes[self.false_label]).item()
        
        similarities = {self.human_label: sim_h, self.false_label: sim_f}

        # Apply dual-threshold logic
        is_human = sim_h >= self.thresholds[self.human_label] and sim_h > sim_f
        is_false = sim_f >= self.thresholds[self.false_label] and sim_f > sim_h
        
        if is_human:
            return self.human_label, similarities
        elif is_false:
            return self.false_label, similarities
        else:
            return self.unknown_label, similarities
    
    def batch_classify(self, features):
        """Classify a batch of feature vectors using dual thresholds.
        
        Args:
            features: Batch of feature vectors (n_samples, feature_dim)
            
        Returns:
            tuple: (predicted_classes_np, all_similarities_np)
                   where all_similarities_np is a dict {label: similarity_array}
        """
        # Convert to tensor if necessary
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # Ensure features are normalized
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarities to human and false prototypes
        proto_h = self.prototypes[self.human_label]
        proto_f = self.prototypes[self.false_label]
        
        sims_h = torch.matmul(features, proto_h) # Shape: [n_samples]
        sims_f = torch.matmul(features, proto_f) # Shape: [n_samples]
        
        # Apply dual-threshold logic batch-wise
        human_thresh = self.thresholds[self.human_label]
        false_thresh = self.thresholds[self.false_label]
        
        is_human_mask = (sims_h >= human_thresh) & (sims_h > sims_f)
        is_false_mask = (sims_f >= false_thresh) & (sims_f > sims_h)
        
        # Initialize predictions as unknown
        predicted_classes = torch.full_like(sims_h, self.unknown_label, dtype=torch.long)
        
        # Assign labels based on masks
        predicted_classes[is_human_mask] = self.human_label
        predicted_classes[is_false_mask] = self.false_label
        
        # Prepare similarity dictionary for return
        all_similarities = {
            self.human_label: sims_h.cpu().numpy(),
            self.false_label: sims_f.cpu().numpy()
        }
        
        return predicted_classes.cpu().numpy(), all_similarities

# Renamed function to reflect focus on dual-threshold evaluation
def evaluate_dual_threshold_osr(classifier, features, labels, plot=False, save_path=None):
    """Evaluate the dual-threshold OSR classifier.
    
    Args:
        classifier: OSR classifier (dual-threshold version)
        features: Test features
        labels: Test labels (should include human=0, false=1, and potentially others for unknown)
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
    
    # Assume known classes are human (0) and false (1)
    known_classes = [classifier.human_label, classifier.false_label]
    
    # Get predictions and similarities dictionary
    pred_labels, similarities_dict = classifier.batch_classify(features)
    
    # --- Calculate Metrics ---
    metrics = {}
    
    # 1. Accuracy on Human samples
    human_mask = (labels == classifier.human_label)
    if np.any(human_mask):
        metrics['human_accuracy'] = accuracy_score(labels[human_mask], pred_labels[human_mask])
    else:
        metrics['human_accuracy'] = np.nan # Or 0.0, depending on desired behavior

    # 2. Accuracy on False samples
    false_mask = (labels == classifier.false_label)
    if np.any(false_mask):
         metrics['false_accuracy'] = accuracy_score(labels[false_mask], pred_labels[false_mask])
    else:
         metrics['false_accuracy'] = np.nan

    # 3. Rejection rate of True Unknowns 
    # (Samples whose true label is NOT human or false)
    unknown_mask = ~np.isin(labels, known_classes)
    if np.any(unknown_mask):
        metrics['unknown_rejection_rate'] = np.mean(pred_labels[unknown_mask] == classifier.unknown_label)
    else:
        metrics['unknown_rejection_rate'] = np.nan # No true unknowns to evaluate

    # 4. Overall Accuracy (excluding true unknowns for classification accuracy)
    known_mask = np.isin(labels, known_classes)
    if np.any(known_mask):
        metrics['overall_known_accuracy'] = accuracy_score(labels[known_mask], pred_labels[known_mask])
    else:
        metrics['overall_known_accuracy'] = np.nan
        
    # --- Plotting ---
    if plot:
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Plot similarity distributions for human/false separately
        plt.figure(figsize=(12, 6))
        
        # Similarities to Human Prototype
        plt.subplot(1, 2, 1)
        sims_h = similarities_dict[classifier.human_label]
        plt.hist(sims_h[human_mask], bins=50, alpha=0.5, label='True Human')
        plt.hist(sims_h[false_mask], bins=50, alpha=0.5, label='True False')
        if np.any(unknown_mask):
             plt.hist(sims_h[unknown_mask], bins=50, alpha=0.5, label='True Unknown')
        plt.axvline(x=classifier.thresholds[classifier.human_label], color='b', linestyle='--', 
                   label=f'Human Thresh = {classifier.thresholds[classifier.human_label]:.2f}')
        plt.xlabel('Similarity to Human Prototype')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Sims vs Human Prototype')

        # Similarities to False Prototype
        plt.subplot(1, 2, 2)
        sims_f = similarities_dict[classifier.false_label]
        plt.hist(sims_f[human_mask], bins=50, alpha=0.5, label='True Human')
        plt.hist(sims_f[false_mask], bins=50, alpha=0.5, label='True False')
        if np.any(unknown_mask):
             plt.hist(sims_f[unknown_mask], bins=50, alpha=0.5, label='True Unknown')
        plt.axvline(x=classifier.thresholds[classifier.false_label], color='r', linestyle='--', 
                   label=f'False Thresh = {classifier.thresholds[classifier.false_label]:.2f}')
        plt.xlabel('Similarity to False Prototype')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Sims vs False Prototype')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_similarity_dist.png")
            plt.close()
        else:
            plt.show()
            
        # Remove ROC curve plot as it was based on max similarity and single threshold
        # # Plot ROC curve
        # plt.figure(figsize=(10, 6))
        # plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}')
        # ... (rest of ROC plot code removed)
    
    return metrics

def parse_args():
    parser = argparse.ArgumentParser('Evaluate Dual-Threshold OSR classifier')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--prototypes', type=str, required=True, help='path to prototypes file (must contain 0 and 1)')
    # Changed threshold argument
    parser.add_argument('--threshold_human', type=float, required=True, help='similarity threshold for human class (0)')
    parser.add_argument('--threshold_false', type=float, required=True, help='similarity threshold for false class (1)')
    # Removed known_classes argument as it's implicit (0, 1)
    # parser.add_argument('--known_classes', type=int, nargs='+', help='known class indices') 
    parser.add_argument('--test_features', type=str, help='path to test features (if not using model)')
    parser.add_argument('--pretrained', type=str, help='pretrained model path (if not using test_features)')
    parser.add_argument('--plot', action='store_true', help='generate plots')
    parser.add_argument('--output_dir', type=str, default='results_dual_thresh', help='output directory') # Changed default
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load prototypes
    with open(args.prototypes, 'rb') as f:
        prototypes = pickle.load(f)
    
    # Ensure prototypes 0 and 1 exist
    if 0 not in prototypes or 1 not in prototypes:
         raise ValueError(f"Prototypes file {args.prototypes} must contain keys for 0 (human) and 1 (false)")
    print(f"Loaded prototypes for classes: {list(prototypes.keys())}")

    # Create threshold dictionary from args
    thresholds_dict = {
        0: args.threshold_human,
        1: args.threshold_false
    }
    print(f"Using thresholds: Human(0)={thresholds_dict[0]:.4f}, False(1)={thresholds_dict[1]:.4f}")
    
    # Create classifier with the threshold dictionary
    classifier = OSRClassifier(prototypes, thresholds_dict=thresholds_dict)
    
    # Get test features and labels
    if args.test_features:
        print(f"Loading test features from: {args.test_features}")
        with open(args.test_features, 'rb') as f:
            data = pickle.load(f)
        test_features = data['features']
        test_labels = data['labels']
    else:
        # Extract features using model
        assert args.pretrained, "Either --test_features or --pretrained must be specified"
        print(f"Extracting features using model: {args.pretrained} and config: {args.cfg}")
        
        # --- Feature Extraction Logic (Simplified) ---
        # Assume PointNeXtFeatureExtractor and build_dataloader_from_cfg work as intended
        # Import config dynamically (handle potential path issues)
        cfg_path = Path(args.cfg)
        sys.path.insert(0, str(cfg_path.parent.parent)) # Add root if cfg is in configs/
        try:
             config = __import__(cfg_path.stem, fromlist=[''])
        except ImportError as e:
             print(f"Error importing config file {args.cfg}: {e}")
             sys.path.pop(0) # Clean up path
             sys.exit(1)
        sys.path.pop(0) # Clean up path

        # Build model
        model = PointNeXtFeatureExtractor(model_cfg=config.model, pretrained_path=args.pretrained)
        model.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(model.device)
        model.eval()

        # Build dataloader (assuming a 'val' or 'test' split exists in config)
        # TODO: User might need to specify which split to use for evaluation
        test_loader, _ = build_dataloader_from_cfg(config.cfg, config.DATA_PATH, batch_size=32, split='val') 
        print(f"Built dataloader for split 'val'. Num samples: {len(test_loader.dataset)}")

        all_features = []
        all_labels = []
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Extracting test features"):
                points = batch_data['points'].to(model.device)
                labels = batch_data['label'].numpy() # Assuming labels are available

                features = model.extract_features(points) # Normalized inside wrapper? Check wrapper. Assume yes for now.
                # If not normalized in wrapper, uncomment:
                # features = F.normalize(features, p=2, dim=1) 

                all_features.append(features.cpu().numpy())
                all_labels.append(labels)

        test_features = np.concatenate(all_features, axis=0)
        test_labels = np.concatenate(all_labels, axis=0)
        print(f"Extracted features shape: {test_features.shape}, labels shape: {test_labels.shape}")
        # --- End Feature Extraction ---

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    save_prefix = os.path.join(args.output_dir, 'eval')

    # Perform evaluation using the updated function
    print("\nStarting evaluation...")
    metrics = evaluate_dual_threshold_osr(classifier, test_features, test_labels, 
                                       plot=args.plot, save_path=save_prefix)
    
    # Print metrics
    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}") # Handle potential NaNs or other types
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Saved metrics to {metrics_path}")

    # Save predictions if needed (optional)
    # preds_path = os.path.join(args.output_dir, 'predictions.pkl')
    # pred_labels, _ = classifier.batch_classify(test_features) # Re-classify to get labels if needed elsewhere
    # with open(preds_path, 'wb') as f:
    #     pickle.dump({'true_labels': test_labels, 'predicted_labels': pred_labels}, f)
    # print(f"Saved predictions to {preds_path}")

if __name__ == "__main__":
    main() 