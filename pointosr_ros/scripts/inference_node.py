#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String # To publish the result
import numpy as np
import torch
import yaml
import message_filters # For synchronizing topics

# It's good practice to add a try-except block for ROS imports
# to allow the script to be run without a ROS environment for testing.
try:
    import ros_numpy
    # Assuming your workspace is sourced, these imports should work.
    from pointnext.dataset.online.inference import PointCloudProcessor
    from pointnext.utils import EasyConfig, load_checkpoint
    from pointnext.model import build_model_from_cfg
    from pointnext.dataset.human.human import HumanDataset # To get class names
except ImportError as e:
    print(f"Error: Could not import required modules. Make sure your ROS workspace is sourced and all dependencies are installed. Details: {e}")
    exit(1)


class InferenceNode:
    """
    A ROS node for running live point cloud inference with PointNext on multiple topics.
    """
    def __init__(self):
        rospy.init_node('pointcloud_inference_node')
        
        # --- Parameters ---
        self.input_topic_prefix = rospy.get_param('~input_topic_prefix', '/cluster_')
        self.output_topic_suffix = rospy.get_param('~output_topic_suffix', '/class')
        cfg_path = rospy.get_param('~cfg_path', 'path/to/your/config.yaml')
        model_path = rospy.get_param('~model_path', 'path/to/your/model.pth')
        num_points = rospy.get_param('~num_points', 2048)
        self.device = rospy.get_param('~device', 'cuda')
        self.topic_check_period = rospy.get_param('~topic_check_period', 5.0) # Check every 5 seconds
        self.sync_slop = rospy.get_param('~sync_slop', 0.2) # Synchronization tolerance in seconds

        # --- Initialization ---
        self.processor = PointCloudProcessor(num_points=num_points, device=self.device)
        rospy.loginfo(f"PointCloud processor initialized for {num_points} points on device '{self.device}'.")

        # --- Load Config and Model ---
        try:
            cfg = EasyConfig()
            cfg.load(cfg_path, recursive=True)
            node_params = rospy.get_param('~')
            cfg.update(node_params)

            self.model = build_model_from_cfg(cfg.model).to(self.device)
            load_checkpoint(self.model, pretrained_path=model_path)
            self.model.eval()
            
            rospy.loginfo(f"Model loaded successfully from {model_path}.")
            self.class_names = cfg.get('classes', HumanDataset.classes)
            rospy.loginfo(f"Using class names: {self.class_names}")

        except Exception as e:
            rospy.logerr(f"Failed to load model or config. Shutting down. Error: {e}")
            rospy.signal_shutdown(f"Model loading failed: {e}")
            return

        # --- Dynamic Topic Management ---
        self.input_topics = []
        self.subscribers = [] # Must hold a reference to subscribers
        self.result_publishers = {}
        self.time_synchronizer = None

        # Call once to initialize, then set up a timer for periodic checks
        self.update_topics() 
        rospy.Timer(rospy.Duration(self.topic_check_period), self.update_topics)
        rospy.loginfo(f"Node initialized. Will check for topics with prefix '{self.input_topic_prefix}' every {self.topic_check_period} seconds.")

    def update_topics(self, event=None):
        """
        Periodically checks for new topics and reconfigures subscribers.
        This makes the node dynamic, allowing it to adapt to topics that
        appear or disappear during runtime.
        """
        try:
            all_topics = rospy.get_published_topics()
        except rospy.ROSException:
            rospy.logwarn("Could not get published topics. Is master available?")
            return

        current_topics = sorted([
            topic for topic, msg_type in all_topics 
            if topic.startswith(self.input_topic_prefix) and msg_type == 'sensor_msgs/PointCloud2'
        ])

        if set(current_topics) == set(self.input_topics):
            return

        rospy.loginfo(f"Topic change detected. Old: {self.input_topics}, New: {current_topics}")
        self.input_topics = current_topics

        if not self.input_topics:
            rospy.logwarn_throttle(30, f"No topics found with prefix '{self.input_topic_prefix}'. Will keep searching.")
            self.subscribers.clear()
            self.time_synchronizer = None
            return

        for topic in self.input_topics:
            if topic not in self.result_publishers:
                publisher_topic = topic + self.output_topic_suffix
                self.result_publishers[topic] = rospy.Publisher(publisher_topic, String, queue_size=10)
                rospy.loginfo(f"Created publisher for new topic: {publisher_topic}")

        # Re-create subscribers and synchronizer
        self.subscribers = [message_filters.Subscriber(topic, PointCloud2) for topic in self.input_topics]
        
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            self.subscribers,
            queue_size=10,
            slop=self.sync_slop
        )
        # Pass a copy of the topics list to the callback to prevent race conditions.
        self.time_synchronizer.registerCallback(self.pointcloud_callback, list(self.input_topics))

        rospy.loginfo(f"Now synchronizing {len(self.input_topics)} topics. Publishing results with suffix '{self.output_topic_suffix}'")

    def pointcloud_callback(self, *args):
        """
        Callback for synchronized point clouds. Processes a batch.
        The topic list is passed as the last argument to avoid race conditions.
        """
        # Extract the topic list and the messages from the arguments
        topic_list = args[-1]
        msgs = args[:-1]

        if not msgs:
            return

        rospy.loginfo_throttle(1.0, f"Received a synchronized batch of {len(msgs)} point clouds.")
        
        batch_pos = []
        batch_x = []
        
        try:
            for i, msg in enumerate(msgs):
                topic_name = topic_list[i]
                
                # 1. Convert PointCloud2 to a NumPy array
                pc_data = ros_numpy.numpify(msg)
                points = np.zeros((len(pc_data), 4), dtype=np.float32)
                points[:, 0] = pc_data['x']
                points[:, 1] = pc_data['y']
                points[:, 2] = pc_data['z']
                points[:, 3] = pc_data['intensity']
                points = points[np.isfinite(points).all(axis=1)]

                if points.shape[0] == 0:
                    rospy.logwarn(f"Received empty point cloud on {topic_name}. Skipping entire batch.")
                    return

                # 2. Process the point cloud (sampling, etc.)
                data = self.processor.process(points)

                batch_pos.append(data['pos'])
                batch_x.append(data['x'])

            if not batch_pos:
                return # All point clouds in batch were empty

            # 3. Stack tensors to create a batch for the model
            pos_tensor = torch.stack(batch_pos, dim=0).to(self.device)
            x_tensor = torch.stack(batch_x, dim=0).to(self.device)

            model_data = {
                'pos': pos_tensor,
                'x': x_tensor.transpose(1, 2) # Model expects (B, C, N)
            }
            
            # 4. Run inference
            with torch.no_grad():
                logits = self.model(model_data)
                pred_indices = torch.argmax(logits, dim=1)
                
                # 5. Publish results for each point cloud in the batch
                predictions_log = []
                for i in range(pred_indices.shape[0]):
                    pred_class_idx = pred_indices[i].item()
                    class_name = self.class_names[pred_class_idx]
                    topic_name = topic_list[i]
                    
                    publisher = self.result_publishers[topic_name]
                    publisher.publish(String(data=class_name))

                    predictions_log.append(f"{topic_name}: '{class_name}'")
                
                if predictions_log:
                    rospy.loginfo_throttle(1.0, f"Batch Predictions: {'; '.join(predictions_log)}")

        except Exception as e:
            rospy.logerr(f"Error processing point cloud batch: {e}")

if __name__ == '__main__':
    try:
        # We need to create the class instance to start the node
        inference_node = InferenceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unhandled exception in node initialization: {e}") 