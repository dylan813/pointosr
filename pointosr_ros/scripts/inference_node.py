#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import numpy as np
import torch
import sys
import os
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from pointosr_ros.msg import PointCloud2Array
from pointosr.core.inference import PointOSRInference
from pointosr.core.processor import PointCloudProcessor

try:
    import ros_numpy
except ImportError as e:
    print(f"Error: Could not import required modules. Make sure your ROS workspace is sourced and all dependencies are installed. Details: {e}")
    exit(1)

class InferenceNode:
    """
    A ROS node for running point cloud inference.
    It subscribes to an aggregated topic of point clouds, processes them in a batch,
    and publishes the classification results.
    """
    def __init__(self):
        rospy.init_node('pointosr_live_inference')

        # --- Parameters ---
        cfg_path = rospy.get_param('~cfg_path', 'path/to/your/config.yaml')
        model_path = rospy.get_param('~model_path', 'path/to/your/model.pth')
        input_topic = rospy.get_param('~input_topic', '/clusters_aggregated')
        num_points = rospy.get_param('~num_points', 2048)
        self.device = rospy.get_param('~device', 'cuda')

        # --- Publishers ---
        self.marker_publisher = rospy.Publisher('/classification_markers', MarkerArray, queue_size=10)

        # --- Load Model ---
        try:
            self.model_inference = PointOSRInference(cfg_path, model_path, device=self.device)
            self.processor = PointCloudProcessor(num_points=num_points)
            rospy.loginfo(f"Model loaded successfully from {model_path}.")
            rospy.loginfo(f"Using class names: {self.model_inference.class_names}")
        except Exception as e:
            rospy.logerr(f"Failed to load model or config. Shutting down. Error: {e}")
            return

        # --- Subscriber ---
        self.subscriber = rospy.Subscriber(
            input_topic,
            PointCloud2Array,
            self.process_batch,
            queue_size=2,
            buff_size=2**24  # 24MB buffer
        )
        rospy.loginfo(f"Subscribed to aggregated topic: {input_topic}")

    def get_cloud_center(self, cloud_array):
        """Calculates the geometric center of a point cloud."""
        return np.mean(cloud_array, axis=0)

    def process_batch(self, msg: PointCloud2Array):
        if not msg.clouds:
            return

        batch_points = []
        original_clouds = []
        for cloud_msg in msg.clouds:
            # Convert ROS PointCloud2 to numpy array
            points = np.frombuffer(cloud_msg.data, dtype=np.float32).reshape(-1, 3)
            # Pre-process for the model (e.g., center, normalize, sample)
            processed_points, _ = self.processor.process(points)
            batch_points.append(processed_points)
            original_clouds.append((points, cloud_msg.header)) # Keep the header

        # Stack into a single tensor for batch inference
        batch_tensor = torch.from_numpy(np.array(batch_points)).to(self.device)

        # --- Run Inference ---
        predictions = self.model_inference.predict(batch_tensor)

        # --- Publish Visualizations ---
        marker_array = MarkerArray()
        for i, (pred_label, (original_cloud, original_header)) in enumerate(zip(predictions, original_clouds)):
            center = self.get_cloud_center(original_cloud)
            
            marker = Marker()
            marker.header = original_header # Use the header from the original cloud
            marker.header.stamp = rospy.Time.now() # Update timestamp to now
            marker.ns = "classification"
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.position.x = center[0]
            marker.pose.position.y = center[1]
            marker.pose.position.z = center[2] + 0.5  # Offset text above the cloud
            marker.pose.orientation.w = 1.0
            marker.scale.z = 0.5  # Text size
            marker.color.a = 1.0  # Must be non-zero
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.text = pred_label
            marker.lifetime = rospy.Duration(0.5) # How long the marker persists
            marker_array.markers.append(marker)

            rospy.loginfo(f"Cloud {i} ({msg.topic_names[i]}): Classified as '{pred_label}' in frame '{original_header.frame_id}'")

        if marker_array.markers:
            self.marker_publisher.publish(marker_array)

if __name__ == '__main__':
    try:
        InferenceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unhandled exception in node initialization: {e}") 