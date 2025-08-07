#!/home/cerlab/miniconda3/envs/pointosr/bin/python
"""
Simple monitoring script for the new classification_batch messages.
This helps debug data loss and verify the batch approach is working.
"""
import rospy
from pointosr_ros.msg import classification_batch
import json
from datetime import datetime

class BatchMonitor:
    def __init__(self):
        rospy.init_node('classification_batch_monitor', anonymous=True)
        
        # Statistics tracking
        self.total_batches = 0
        self.total_clusters_received = 0
        self.total_human_detections = 0
        self.total_false_detections = 0
        self.total_processing_errors = 0
        
        # Data loss tracking
        self.expected_vs_received = []
        self.processing_times = []
        
        self.subscriber = rospy.Subscriber(
            '/classified_clusters', 
            classification_batch, 
            self.batch_callback
        )
        
        rospy.loginfo("BatchMonitor started. Listening for classification_batch messages on /classified_clusters")
        
        # Print statistics every 30 seconds
        rospy.Timer(rospy.Duration(30), self.print_statistics)
        
    def batch_callback(self, msg):
        """Process incoming batch messages and track statistics"""
        self.total_batches += 1
        
        # Extract key metrics
        stamp = msg.header.stamp
        total_input = msg.total_input_clusters
        total_processed = msg.total_processed_clusters
        human_count = msg.human_count
        false_count = msg.false_count
        error_count = msg.processing_errors
        processing_time = msg.processing_time_sec
        
        # Update statistics
        self.total_clusters_received += total_processed
        self.total_human_detections += human_count
        self.total_false_detections += false_count
        self.total_processing_errors += error_count
        self.processing_times.append(processing_time)
        
        # Track data loss
        data_loss = total_input - total_processed
        self.expected_vs_received.append({
            'timestamp': stamp.to_sec(),
            'expected': total_input,
            'received': total_processed,
            'data_loss': data_loss
        })
        
        # Log batch summary
        rospy.loginfo(f"Batch {self.total_batches}: "
                     f"Expected={total_input}, Received={total_processed}, "
                     f"Human={human_count}, False={false_count}, "
                     f"Errors={error_count}, Time={processing_time:.3f}s")
        
        # Alert on data loss
        if data_loss > 0:
            rospy.logwarn(f"DATA LOSS DETECTED! Expected {total_input} clusters, "
                         f"but only processed {total_processed}. Loss: {data_loss}")
        
        # Alert on processing errors
        if error_count > 0:
            rospy.logwarn(f"PROCESSING ERRORS: {error_count} clusters failed to process")
            for i, log_entry in enumerate(msg.processing_log):
                if "error" in log_entry.lower() or "failed" in log_entry.lower():
                    rospy.logwarn(f"  Error detail: {log_entry}")
        
        # Log detailed cluster info if debug enabled
        if rospy.get_param('~detailed_logging', False):
            for cluster in msg.classified_clusters:
                rospy.logdebug(f"  Cluster {cluster.original_cluster_index}: "
                              f"class='{cluster.class_name}', "
                              f"conf={cluster.confidence:.3f}, "
                              f"success={cluster.processing_success}")
    
    def print_statistics(self, event):
        """Print comprehensive statistics"""
        if self.total_batches == 0:
            rospy.loginfo("No batches received yet...")
            return
            
        avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        total_data_loss = sum(item['data_loss'] for item in self.expected_vs_received)
        
        rospy.loginfo("="*60)
        rospy.loginfo("CLASSIFICATION BATCH STATISTICS")
        rospy.loginfo("="*60)
        rospy.loginfo(f"Total Batches Processed: {self.total_batches}")
        rospy.loginfo(f"Total Clusters Received: {self.total_clusters_received}")
        rospy.loginfo(f"Total Human Detections: {self.total_human_detections}")
        rospy.loginfo(f"Total False Detections: {self.total_false_detections}")
        rospy.loginfo(f"Total Processing Errors: {self.total_processing_errors}")
        rospy.loginfo(f"Total Data Loss: {total_data_loss} clusters")
        rospy.loginfo(f"Average Processing Time: {avg_processing_time:.3f}s")
        
        if total_data_loss > 0:
            rospy.logwarn(f"⚠️  DATA LOSS DETECTED: {total_data_loss} clusters lost across {self.total_batches} batches")
        else:
            rospy.loginfo("✅ No data loss detected!")
            
        rospy.loginfo("="*60)

if __name__ == '__main__':
    try:
        BatchMonitor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
