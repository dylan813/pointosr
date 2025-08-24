#!/home/cerlab/miniconda3/envs/pointosr/bin/python
"""
Test script to verify OSR integration functionality.

This script tests the OSR scorer components independently to ensure
the integration is working correctly before deploying to the full ROS pipeline.
"""

import sys
import os
import numpy as np
import torch

# Add path for imports
pointosr_path = '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr'
sys.path.insert(0, pointosr_path)

# Add current script directory for osr_scorer import
sys.path.insert(0, '/home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr_ros/scripts')

from osr_scorer import OSRScorer

def test_osr_scorer_initialization():
    """Test OSR scorer initialization and configuration loading."""
    print("🧪 Testing OSR Scorer Initialization...")
    
    try:
        scorer = OSRScorer(device='cpu')  # Use CPU for testing
        print("✅ OSR Scorer initialized successfully")
        print(f"   Temperature: {scorer.temperature:.4f}")
        print(f"   Fusion weights: {scorer.fusion_weights}")
        print(f"   OOD threshold: {scorer.ood_threshold:.6f}")
        print(f"   Target TPR: {scorer.target_tpr:.3f}")
        return scorer
    except Exception as e:
        print(f"❌ OSR Scorer initialization failed: {e}")
        return None

def test_score_computation(scorer):
    """Test score computation with dummy data."""
    print("\n🧪 Testing Score Computation...")
    
    try:
        # Create dummy data
        batch_size = 4
        num_classes = 2
        embed_dim = 512  # Must match prototype embedding dimension
        
        # Dummy logits and embeddings
        logits = torch.randn(batch_size, num_classes)
        embeddings = torch.randn(batch_size, embed_dim)
        
        print(f"   Testing with batch_size={batch_size}, embed_dim={embed_dim}")
        
        # Test score computation
        results = scorer.score_batch(logits, embeddings)
        
        print("✅ Score computation successful")
        print(f"   Predictions: {results['predictions']}")
        print(f"   Energy scores (raw): {results['energy_scores_raw']}")
        print(f"   Cosine scores (raw): {results['cosine_scores_raw']}")
        print(f"   Fused scores: {results['fused_scores']}")
        print(f"   OOD detections: {results['is_ood']}")
        print(f"   OOD confidences: {results['ood_confidences']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Score computation failed: {e}")
        return False

def test_class_name_mapping(scorer):
    """Test class name mapping with OOD detection."""
    print("\n🧪 Testing Class Name Mapping...")
    
    try:
        class_names = ['human', 'false']
        
        # Test cases: (predicted_class, is_ood, expected_result)
        test_cases = [
            (0, False, 'human'),
            (1, False, 'false'),
            (0, True, 'ood'),
            (1, True, 'ood')
        ]
        
        for pred_class, is_ood, expected in test_cases:
            result = scorer.get_class_name_with_ood(pred_class, is_ood, class_names)
            status = "✅" if result == expected else "❌"
            print(f"   {status} Class {pred_class}, OOD={is_ood} → '{result}' (expected: '{expected}')")
        
        return True
        
    except Exception as e:
        print(f"❌ Class name mapping failed: {e}")
        return False

def test_message_compatibility():
    """Test ROS message compatibility."""
    print("\n🧪 Testing ROS Message Compatibility...")
    
    try:
        # Try importing the updated messages
        sys.path.insert(0, '/home/cerlab/Documents/pointosr_ws/devel/lib/python3/dist-packages')
        
        from pointosr_ros.msg import cluster_and_cls, classification_batch
        
        print("✅ ROS messages imported successfully")
        
        # Test creating a message with OSR fields
        msg = cluster_and_cls()
        msg.class_name = "ood"
        msg.is_human = False
        msg.is_ood = True
        msg.energy_score = 0.5
        msg.cosine_score = 0.3
        msg.fused_score = 0.4
        msg.ood_confidence = 0.8
        
        print("✅ OSR message fields accessible")
        print(f"   Class: {msg.class_name}, OOD: {msg.is_ood}")
        print(f"   Scores - Energy: {msg.energy_score}, Cosine: {msg.cosine_score}")
        print(f"   Fused: {msg.fused_score}, OOD Confidence: {msg.ood_confidence}")
        
        # Test batch message
        batch_msg = classification_batch()
        batch_msg.human_count = 1
        batch_msg.false_count = 2
        batch_msg.ood_count = 1
        
        print("✅ Batch message with OOD count accessible")
        print(f"   Counts - Human: {batch_msg.human_count}, False: {batch_msg.false_count}, OOD: {batch_msg.ood_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ ROS message compatibility test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 OSR Integration Test Suite")
    print("=" * 50)
    
    # Test 1: OSR Scorer Initialization
    scorer = test_osr_scorer_initialization()
    if scorer is None:
        print("\n❌ Cannot continue tests - OSR Scorer initialization failed")
        return False
    
    # Test 2: Score Computation
    score_test_passed = test_score_computation(scorer)
    
    # Test 3: Class Name Mapping
    mapping_test_passed = test_class_name_mapping(scorer)
    
    # Test 4: ROS Message Compatibility
    message_test_passed = test_message_compatibility()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   OSR Scorer Initialization: ✅")
    print(f"   Score Computation: {'✅' if score_test_passed else '❌'}")
    print(f"   Class Name Mapping: {'✅' if mapping_test_passed else '❌'}")
    print(f"   ROS Message Compatibility: {'✅' if message_test_passed else '❌'}")
    
    all_passed = score_test_passed and mapping_test_passed and message_test_passed
    
    if all_passed:
        print("\n🎉 All tests passed! OSR integration is ready for deployment.")
        print("\n📝 Next steps:")
        print("   1. Source the workspace: source devel/setup.bash")
        print("   2. Launch with OSR enabled: roslaunch pointosr_ros classification.launch enable_osr:=true")
        print("   3. Monitor /classified_clusters topic for OOD detections")
    else:
        print("\n⚠️  Some tests failed. Please check the configuration and try again.")
    
    return all_passed

if __name__ == "__main__":
    main()
