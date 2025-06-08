#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String, Header # To publish the result and for the trigger
import numpy as np
import torch
import yaml
import threading # For locking the buffer

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
    A ROS node for running point cloud inference, triggered by a 'frame done' topic.
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
        max_cluster_topics = rospy.get_param('~max_cluster_topics', 30)
        trigger_topic = rospy.get_param('~trigger_topic', '/motion_detector/segmentation_done')

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

        # --- Frame Buffer and Synchronization ---
        self.frame_buffer = {}
        self.buffer_lock = threading.Lock()
        
        # --- Publishers and Subscribers ---
        self.result_publishers = {}
        self.cluster_subscribers = []

        rospy.loginfo(f"Setting up subscribers for up to {max_cluster_topics} cluster topics with prefix '{self.input_topic_prefix}'.")
        for i in range(max_cluster_topics):
            topic_name = f"{self.input_topic_prefix}{i}"
            
            # Create publisher for this potential topic ahead of time
            publisher_topic = topic_name + self.output_topic_suffix
            self.result_publishers[topic_name] = rospy.Publisher(publisher_topic, String, queue_size=10)

            # Subscribe to the cluster topic
            # Use a lambda with a default argument to capture the topic_name correctly
            sub = rospy.Subscriber(topic_name, PointCloud2, 
                                   lambda msg, tn=topic_name: self._cluster_callback(msg, tn))
            self.cluster_subscribers.append(sub)

        # Subscriber for the trigger message
        self.trigger_subscriber = rospy.Subscriber(trigger_topic, Header, self._trigger_callback)
        rospy.loginfo(f"Node initialized. Listening for trigger on '{trigger_topic}'.")

    def _cluster_callback(self, msg, topic_name):
        """
        Callback for individual cluster topics. Buffers the received messages.
        """
        with self.buffer_lock:
            stamp = msg.header.stamp
            if stamp not in self.frame_buffer:
                self.frame_buffer[stamp] = {}
            self.frame_buffer[stamp][topic_name] = msg
            rospy.logdebug(f"Buffered cluster from {topic_name} for stamp {stamp}.")

    def _trigger_callback(self, header_msg):
        """
        Callback for the trigger topic. Processes a complete frame.
        """
        stamp = header_msg.stamp
        rospy.logdebug(f"Trigger received for stamp {stamp}.")
        
        batch_data = None
        with self.buffer_lock:
            if stamp in self.frame_buffer:
                batch_data = self.frame_buffer.pop(stamp)
            else:
                rospy.logwarn_throttle(5.0, f"Trigger for stamp {stamp} received, but no clusters were buffered. "
                                            "This could be due to message drops or network delays.")
                return

        if not batch_data:
            rospy.logwarn(f"No data in batch for stamp {stamp} despite trigger.")
            return
        
        # The batch_data is a dict of {topic_name: msg}. We want the messages and topic names.
        topic_list = list(batch_data.keys())
        msgs = [batch_data[tn] for tn in topic_list]
        
        rospy.loginfo(f"Processing batch of {len(msgs)} clusters for stamp {stamp}.")
        self._process_batch(msgs, topic_list, stamp)

    def _process_batch(self, msgs, topic_list, stamp):
        """
        Processes a batch of synchronized point clouds.
        """
        if not msgs:
            return

        batch_pos = []
        batch_x = []
        valid_topic_list = [] # For handling empty pointclouds in a batch
        
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
                    rospy.logwarn(f"Received empty point cloud on {topic_name} in batch {stamp}. Skipping this point cloud.")
                    continue # Skip this one, but process the rest of the batch

                # 2. Process the point cloud (sampling, etc.)
                data = self.processor.process(points)

                batch_pos.append(data['pos'])
                batch_x.append(data['x'])
                valid_topic_list.append(topic_name)

            if not batch_pos:
                rospy.logwarn(f"All point clouds in batch for stamp {stamp} were empty or invalid.")
                return 

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
                    # This relies on the order being preserved.
                    topic_name = valid_topic_list[i] 
                    
                    publisher = self.result_publishers[topic_name]
                    publisher.publish(String(data=class_name))

                    predictions_log.append(f"{topic_name}: '{class_name}'")
                
                if predictions_log:
                    rospy.loginfo(f"Batch Predictions for stamp {stamp}: {'; '.join(predictions_log)}")

        except Exception as e:
            rospy.logerr(f"Error processing point cloud batch for stamp {stamp}: {e}")

if __name__ == '__main__':
    try:
        # We need to create the class instance to start the node
        inference_node = InferenceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unhandled exception in node initialization: {e}") 