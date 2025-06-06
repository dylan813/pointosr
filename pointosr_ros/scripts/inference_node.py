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
        # Get parameters from the ROS parameter server, with defaults
        input_topic_prefix = rospy.get_param('~input_topic_prefix', '/cluster_')
        output_topic_suffix = rospy.get_param('~output_topic_suffix', '/class')
        cfg_path = rospy.get_param('~cfg_path', 'path/to/your/config.yaml')
        model_path = rospy.get_param('~model_path', 'path/to/your/model.pth')
        num_points = rospy.get_param('~num_points', 2048)
        self.device = rospy.get_param('~device', 'cuda')

        # --- Discover topics ---
        rospy.loginfo("Searching for topics...")
        # A short delay can help ensure all topics are discovered after connecting to master.
        rospy.sleep(2.0)
        all_topics = rospy.get_published_topics()
        self.input_topics = sorted([
            topic for topic, msg_type in all_topics 
            if topic.startswith(input_topic_prefix) and msg_type == 'sensor_msgs/PointCloud2'
        ])

        if not self.input_topics:
            rospy.logfatal(f"No topics found with prefix '{input_topic_prefix}'. Shutting down.")
            rospy.signal_shutdown("No input topics found.")
            return
            
        rospy.loginfo(f"Found {len(self.input_topics)} topics: {self.input_topics}")
        
        # --- Initialization ---
        self.processor = PointCloudProcessor(num_points=num_points, device=self.device)
        rospy.loginfo(f"PointCloud processor initialized for {num_points} points on device '{self.device}'.")

        # --- Load Config and Model ---
        try:
            cfg = EasyConfig()
            cfg.load(cfg_path, recursive=True)
            # cfg.update(rospy.get_param_names()) # This is too broad and can cause conflicts.
            
            # More safely, get only the parameters for this node.
            # ROS launch files put params in a private namespace.
            node_params = rospy.get_param('~')
            cfg.update(node_params)

            # Build the model from the configuration file
            self.model = build_model_from_cfg(cfg.model).to(self.device)
            
            # Load the trained weights
            load_checkpoint(self.model, pretrained_path=model_path)
            self.model.eval() # Set the model to evaluation mode
            
            rospy.loginfo(f"Model loaded successfully from {model_path}.")

            # Store class names for publishing
            self.class_names = cfg.get('classes', HumanDataset.classes) # Fallback to HumanDataset.classes
            rospy.loginfo(f"Using class names: {self.class_names}")

        except Exception as e:
            rospy.logerr(f"Failed to load model or config. Shutting down. Error: {e}")
            rospy.signal_shutdown(f"Model loading failed: {e}")
            return

        # --- Publisher and Subscriber ---
        self.result_publishers = {
            topic: rospy.Publisher(topic + output_topic_suffix, String, queue_size=10)
            for topic in self.input_topics
        }

        subscribers = [message_filters.Subscriber(topic, PointCloud2) for topic in self.input_topics]
        
        # Synchronize topics by approximate timestamp
        # slop: allowed time difference between messages
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            subscribers,
            queue_size=10,
            slop=0.2 # 200ms tolerance, adjust as needed
        )
        self.time_synchronizer.registerCallback(self.pointcloud_callback)

        rospy.loginfo(f"Synchronizing {len(self.input_topics)} topics. Publishing results with suffix '{output_topic_suffix}'")


    def pointcloud_callback(self, *msgs):
        """
        Callback for synchronized point clouds. Processes a batch.
        """
        if not msgs:
            return

        rospy.loginfo_throttle(1.0, f"Received a synchronized batch of {len(msgs)} point clouds.")
        
        batch_pos = []
        batch_x = []
        identifiers = []
        # The order of messages corresponds to the order of topics in self.input_topics
        
        try:
            for i, msg in enumerate(msgs):
                topic_name = self.input_topics[i]
                
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
                identifier = msg.header.stamp.to_sec()
                data = self.processor.process(points, identifier=identifier)

                batch_pos.append(data['pos'])
                batch_x.append(data['x'])
                identifiers.append(identifier)

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
                for i in range(pred_indices.shape[0]):
                    pred_class_idx = pred_indices[i].item()
                    class_name = self.class_names[pred_class_idx]
                    topic_name = self.input_topics[i]
                    identifier = identifiers[i]
                    
                    rospy.loginfo_throttle(1.0, f"Prediction on {topic_name} (frame {identifier}): '{class_name}'")
                    
                    publisher = self.result_publishers[topic_name]
                    publisher.publish(String(data=class_name))

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