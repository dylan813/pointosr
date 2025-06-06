#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String # To publish the result
import numpy as np
import torch
import yaml

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
    A ROS node for running live point cloud inference with PointNext.
    """
    def __init__(self):
        rospy.init_node('pointcloud_inference_node')
        
        # --- Parameters ---
        # Get parameters from the ROS parameter server, with defaults
        input_topic = rospy.get_param('~input_topic', '/velodyne_points')
        output_topic = rospy.get_param('~output_topic', '~/pointcloud_class')
        cfg_path = rospy.get_param('~cfg_path', 'path/to/your/config.yaml')
        model_path = rospy.get_param('~model_path', 'path/to/your/model.pth')
        num_points = rospy.get_param('~num_points', 2048)
        self.device = rospy.get_param('~device', 'cuda')
        
        # --- Initialization ---
        self.processor = PointCloudProcessor(num_points=num_points, device=self.device)
        rospy.loginfo(f"PointCloud processor initialized for {num_points} points on device '{self.device}'.")

        # --- Load Config and Model ---
        try:
            cfg = EasyConfig()
            cfg.load(cfg_path, recursive=True)
            cfg.update(rospy.get_param_names()) # Allow ROS params to override config
            
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
        self.result_publisher = rospy.Publisher(output_topic, String, queue_size=10)
        self.subscriber = rospy.Subscriber(
            input_topic, 
            PointCloud2, 
            self.pointcloud_callback,
            queue_size=1,
            buff_size=2**24 # Approx 16MB buffer
        )
        rospy.loginfo(f"Subscribed to {input_topic} and publishing results to {output_topic}")


    def pointcloud_callback(self, msg):
        """
        Callback function for the PointCloud2 subscriber.
        """
        rospy.loginfo(f"Received point cloud with timestamp: {msg.header.stamp.to_sec()}")
        
        try:
            # 1. Convert PointCloud2 to a NumPy array
            pc_data = ros_numpy.numpify(msg)
            points = np.zeros((len(pc_data), 4), dtype=np.float32)
            points[:, 0] = pc_data['x']
            points[:, 1] = pc_data['y']
            points[:, 2] = pc_data['z']
            points[:, 3] = pc_data['intensity']
            points = points[np.isfinite(points).all(axis=1)]

            if points.shape[0] == 0:
                rospy.logwarn("Received an empty or all-NaN point cloud. Skipping.")
                return

            # 2. Process the point cloud using our processor
            identifier = msg.header.stamp.to_sec()
            data = self.processor.process(points, identifier=identifier)

            # 3. Prepare data for the model
            # Add a batch dimension and move to the correct device
            data['pos'] = data['pos'].unsqueeze(0).to(self.device)
            
            # The model expects features in shape (B, C, N), so we transpose
            x_features = data['x'].unsqueeze(0).transpose(1, 2).to(self.device)
            data['x'] = x_features
            
            # 4. Run inference
            with torch.no_grad():
                logits = self.model(data)
                pred_class_idx = torch.argmax(logits, dim=1).item()
                class_name = self.class_names[pred_class_idx]
                
                rospy.loginfo(f"Prediction for frame {identifier}: '{class_name}' (Class index: {pred_class_idx})")
                
                # 5. Publish the result
                self.result_publisher.publish(String(data=class_name))

        except Exception as e:
            rospy.logerr(f"Error processing point cloud: {e}")

if __name__ == '__main__':
    try:
        # We need to create the class instance to start the node
        inference_node = InferenceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unhandled exception in node initialization: {e}") 