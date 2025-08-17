#!/home/cerlab/miniconda3/envs/pointosr/bin/python
"""
OSR Pipeline Diagnostics

This script monitors the /classified_clusters topic and provides detailed analysis
of OSR scores to help understand why everything is being classified as OOD.
"""

import rospy
import sys
import numpy as np
from collections import defaultdict, deque
from pointosr_ros.msg import classification_batch, cluster_and_cls

class OSRDiagnostics:
    def __init__(self):
        rospy.init_node('osr_diagnostics', anonymous=True)
        
        # Statistics tracking
        self.batch_count = 0
        self.total_clusters = 0
        self.class_counts = defaultdict(int)
        self.score_history = {
            'energy_raw': deque(maxlen=1000),
            'cosine_raw': deque(maxlen=1000),
            'fused': deque(maxlen=1000),
            'ood_confidence': deque(maxlen=1000)
        }
        self.recent_batches = deque(maxlen=10)
        
        # Subscribe to classification results
        self.sub = rospy.Subscriber('/classified_clusters', classification_batch, self.batch_callback)
        
        rospy.loginfo("OSR Diagnostics started. Monitoring /classified_clusters...")
        print("\n" + "="*80)
        print("🔍 OSR PIPELINE DIAGNOSTICS")
        print("="*80)
        print("Monitoring real-time OSR classifications...")
        print("Press Ctrl+C to see summary statistics\n")
    
    def batch_callback(self, msg):
        """Process each batch of classifications."""
        self.batch_count += 1
        batch_stats = {
            'timestamp': msg.header.stamp.to_sec(),
            'total_input': msg.total_input_clusters,
            'processed': msg.total_processed_clusters,
            'human': msg.human_count,
            'false': msg.false_count,
            'ood': msg.ood_count,
            'errors': msg.processing_errors,
            'processing_time': msg.processing_time_sec,
            'clusters': []
        }
        
        # Process individual clusters
        for cluster in msg.classified_clusters:
            if cluster.processing_success:
                self.total_clusters += 1
                self.class_counts[cluster.class_name] += 1
                
                # Collect scores
                self.score_history['energy_raw'].append(cluster.energy_score)
                self.score_history['cosine_raw'].append(cluster.cosine_score)
                self.score_history['fused'].append(cluster.fused_score)
                self.score_history['ood_confidence'].append(cluster.ood_confidence)
                
                cluster_info = {
                    'class': cluster.class_name,
                    'is_ood': cluster.is_ood,
                    'confidence': cluster.confidence,
                    'energy_raw': cluster.energy_score,
                    'cosine_raw': cluster.cosine_score,
                    'fused': cluster.fused_score,
                    'ood_conf': cluster.ood_confidence
                }
                batch_stats['clusters'].append(cluster_info)
        
        self.recent_batches.append(batch_stats)
        
        # Print real-time summary every 5 batches
        if self.batch_count % 5 == 0:
            self.print_current_stats()
    
    def print_current_stats(self):
        """Print current statistics."""
        if not self.recent_batches:
            return
            
        recent = list(self.recent_batches)[-5:]  # Last 5 batches
        
        print(f"\n📊 Batch {self.batch_count} Summary (last 5 batches):")
        print(f"   Total clusters processed: {self.total_clusters}")
        print(f"   Class distribution: {dict(self.class_counts)}")
        
        if self.score_history['fused']:
            fused_scores = list(self.score_history['fused'])[-50:]  # Last 50 scores
            energy_scores = list(self.score_history['energy_raw'])[-50:]
            cosine_scores = list(self.score_history['cosine_raw'])[-50:]
            
            print(f"   Recent score ranges:")
            print(f"     Energy (raw): [{min(energy_scores):.3f}, {max(energy_scores):.3f}] (mean: {np.mean(energy_scores):.3f})")
            print(f"     Cosine (raw): [{min(cosine_scores):.3f}, {max(cosine_scores):.3f}] (mean: {np.mean(cosine_scores):.3f})")
            print(f"     Fused:        [{min(fused_scores):.3f}, {max(fused_scores):.3f}] (mean: {np.mean(fused_scores):.3f})")
            
            # Show some individual examples
            if recent:
                latest_batch = recent[-1]
                if latest_batch['clusters']:
                    print(f"   Latest batch examples:")
                    for i, cluster in enumerate(latest_batch['clusters'][:3]):  # Show first 3
                        print(f"     Cluster {i+1}: {cluster['class']} (fused: {cluster['fused']:.3f}, "
                             f"energy: {cluster['energy_raw']:.3f}, cosine: {cluster['cosine_raw']:.3f})")
    
    def print_detailed_analysis(self):
        """Print detailed analysis of all collected data."""
        print("\n" + "="*80)
        print("📈 DETAILED OSR ANALYSIS")
        print("="*80)
        
        print(f"Total batches processed: {self.batch_count}")
        print(f"Total clusters processed: {self.total_clusters}")
        print(f"Class distribution: {dict(self.class_counts)}")
        
        if not self.score_history['fused']:
            print("❌ No score data collected!")
            return
        
        # Score statistics
        scores = {
            'Energy (raw)': list(self.score_history['energy_raw']),
            'Cosine (raw)': list(self.score_history['cosine_raw']),
            'Fused': list(self.score_history['fused']),
            'OOD Confidence': list(self.score_history['ood_confidence'])
        }
        
        print(f"\n📊 Score Statistics:")
        for score_name, values in scores.items():
            if values:
                print(f"   {score_name}:")
                print(f"     Range: [{min(values):.4f}, {max(values):.4f}]")
                print(f"     Mean:  {np.mean(values):.4f}")
                print(f"     Std:   {np.std(values):.4f}")
                print(f"     Percentiles: 10%={np.percentile(values, 10):.4f}, "
                     f"50%={np.percentile(values, 50):.4f}, 90%={np.percentile(values, 90):.4f}")
        
        # Current OSR configuration (from known values)
        print(f"\n⚙️ Current OSR Configuration:")
        print(f"   OOD Threshold: 0.076659 (from fusion config)")
        print(f"   Target TPR: 99.0%")
        print(f"   Fusion weights: [0.05 energy, 0.95 cosine]")
        
        # Analysis
        fused_scores = scores['Fused']
        threshold = 0.076659
        above_threshold = sum(1 for score in fused_scores if score >= threshold)
        below_threshold = len(fused_scores) - above_threshold
        
        print(f"\n🎯 Threshold Analysis:")
        print(f"   Samples above threshold (ID): {above_threshold} ({above_threshold/len(fused_scores)*100:.1f}%)")
        print(f"   Samples below threshold (OOD): {below_threshold} ({below_threshold/len(fused_scores)*100:.1f}%)")
        print(f"   Max fused score seen: {max(fused_scores):.4f}")
        print(f"   Current threshold: {threshold:.4f}")
        
        if max(fused_scores) < threshold:
            print(f"   ⚠️  ALL samples are below threshold! Consider lowering threshold.")
            suggested_threshold = np.percentile(fused_scores, 95)
            print(f"   💡 Suggested threshold (95th percentile): {suggested_threshold:.4f}")
        
        # Recent examples
        if self.recent_batches:
            print(f"\n📝 Recent Examples:")
            latest_batch = list(self.recent_batches)[-1]
            for i, cluster in enumerate(latest_batch['clusters'][:5]):
                ood_status = "OOD" if cluster['is_ood'] else "ID"
                print(f"   {i+1}. {cluster['class']} ({ood_status}): "
                     f"fused={cluster['fused']:.4f}, energy={cluster['energy_raw']:.3f}, "
                     f"cosine={cluster['cosine_raw']:.3f}")
    
    def run(self):
        """Run the diagnostics."""
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping diagnostics...")
            self.print_detailed_analysis()
            
            # Recommendations
            print(f"\n💡 RECOMMENDATIONS:")
            if self.class_counts.get('ood', 0) == self.total_clusters:
                print("   🔥 ISSUE: Everything is classified as OOD!")
                print("   📝 Possible causes:")
                print("      1. Threshold is too high for current data distribution")
                print("      2. Model/prototypes were trained on different data")
                print("      3. Score normalization calibrated on different dataset")
                print("   🛠️  Suggested actions:")
                print("      1. Lower the OOD threshold in fusion_config.json")
                print("      2. Re-calibrate OSR on current dataset")
                print("      3. Check if model and OSR configs match")
            
            print(f"\n📋 To adjust threshold, edit:")
            print(f"   /home/cerlab/Documents/pointosr_ws/src/pointosr/pointosr/osr/calibration/fusion_config.json")
            print(f"   Change 'fused_threshold' value and restart the node")

def main():
    try:
        diagnostics = OSRDiagnostics()
        diagnostics.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
