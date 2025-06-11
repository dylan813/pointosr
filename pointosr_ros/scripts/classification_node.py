#!/home/cerlab/miniconda3/envs/pointosr/bin/python
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import numpy as np
import torch
import threading
from collections import OrderedDict
import ros_numpy
from pointnext.utils import EasyConfig, load_checkpoint
from pointnext.model import build_model_from_cfg
from pointnext.dataset.online.online_classification import OnlineDataloader

class ClassificationNode:
    def __init__(self):
        rospy.init_node('pointosr_live_classification', anonymous=True)
        
        trigger_topic = rospy.get_param('~trigger_topic', '/motion_detector/cluster_batch')
        self.input_topic_prefix = rospy.get_param('~input_topic_prefix', '/cluster_')
        self.filtered_topic_prefix = rospy.get_param('~filtered_topic_prefix', '/filt_cluster_')
        max_cluster_topics = rospy.get_param('~max_cluster_topics', 30)
        self.buffer_timeout = rospy.get_param('~buffer_timeout', 10.0)

        model_path = rospy.get_param('~model_path')
        cfg_path = rospy.get_param('~cfg_path')
        device = rospy.get_param('~device', 'cuda')
        num_points = rospy.get_param('~num_points', 2048)

        try:
            self.processor = OnlineDataloader(num_points=num_points, device=device)
            rospy.loginfo(f"Processor initialized for {num_points} points on '{device}'.")
            
            cfg = EasyConfig()
            cfg.load(cfg_path, recursive=True)
            cfg.update(rospy.get_param('~'))
            self.model = build_model_from_cfg(cfg.model).to(device)
            load_checkpoint(self.model, pretrained_path=model_path)
            self.model.eval()
            self.device = device
            rospy.loginfo(f"Model loaded from {model_path}.")
            self.class_names = cfg.get('classes', OnlineDataloader.classes)
            rospy.loginfo(f"Using classes: {self.class_names}")
        except Exception as e:
            rospy.logerr(f"Failed to load model/config: {e}")
            rospy.signal_shutdown("Model loading failed.")
            return

        self.frame_buffer = OrderedDict()
        self.trigger_buffer = OrderedDict()
        self.buffer_lock = threading.Lock()
        
        # Dictionary for lazy-initialised publishers of filtered clusters.
        self.filtered_publishers = {}
        
        # Add completion trigger publisher
        self.completion_trigger_pub = rospy.Publisher(
            trigger_topic.replace('cluster_batch', 'filtered_complete'), 
            Header, queue_size=10
        )
        
        rospy.loginfo(f"Subscribing to up to {max_cluster_topics} topics with prefix '{self.input_topic_prefix}'.")     #might remove later these good for sanity
        rospy.loginfo(f"Will publish individual filtered clusters with prefix '{self.filtered_topic_prefix}'.")
        self.cluster_subscribers = [
            self._setup_subscriber(i) for i in range(max_cluster_topics)
        ]

        self.trigger_subscriber = rospy.Subscriber(trigger_topic, Header, self._trigger_callback)
        rospy.loginfo(f"Node ready. Listening for trigger on '{trigger_topic}'.")

    def _setup_subscriber(self, index):
        topic_name = f"{self.input_topic_prefix}{index}"
        return rospy.Subscriber(
            topic_name, PointCloud2,
            lambda msg, tn=topic_name: self._cluster_callback(msg, tn)
        )

    def _cluster_callback(self, msg, topic_name):
        with self.buffer_lock:
            stamp = msg.header.stamp
            now = rospy.Time.now()
            
            if stamp not in self.frame_buffer:
                self.frame_buffer[stamp] = {'messages': {}, 'arrival_time': now}
            
            self.frame_buffer[stamp]['messages'][topic_name] = msg
            
            if stamp in self.trigger_buffer:
                expected_count = self.trigger_buffer[stamp]['expected_count']
                if len(self.frame_buffer[stamp]['messages']) == expected_count:
                    self._process_frame_if_complete(stamp)

    def _trigger_callback(self, header_msg):
        with self.buffer_lock:
            stamp = header_msg.stamp
            expected_count = header_msg.seq
            now = rospy.Time.now()
            
            if expected_count == 0:
                rospy.logdebug(f"Trigger for stamp {stamp} received with 0 expected clusters, ignoring.")
                return

            self.trigger_buffer[stamp] = {'expected_count': expected_count, 'arrival_time': now}

            if stamp in self.frame_buffer:
                if len(self.frame_buffer[stamp]['messages']) == expected_count:
                    self._process_frame_if_complete(stamp)
            
            self._cleanup_buffers()

    def _process_frame_if_complete(self, stamp):
        rospy.logdebug(f"Frame for stamp {stamp} is complete. Processing.")
        
        batch_data_dict = self.frame_buffer.pop(stamp)['messages']
        self.trigger_buffer.pop(stamp)
        
        self.buffer_lock.release()
        
        try:
            topic_list = list(batch_data_dict.keys())
            msgs = [batch_data_dict[tn] for tn in topic_list]
            
            rospy.logdebug(f"Processing batch of {len(msgs)} clusters for stamp {stamp}.")
            self._process_batch(msgs, topic_list, stamp)
        finally:
            self.buffer_lock.acquire()

    def _cleanup_buffers(self):
        now = rospy.Time.now()
        
        for s, data in list(self.frame_buffer.items()):
            if (now - data['arrival_time']).to_sec() > self.buffer_timeout:
                expected = self.trigger_buffer.get(s, {}).get('expected_count', 'N/A')
                rospy.logwarn(f"Timing out stale frame {s} (recv: {len(data['messages'])}, expect: {expected}).")
                del self.frame_buffer[s]
                if s in self.trigger_buffer:
                    del self.trigger_buffer[s]
        
        for s, data in list(self.trigger_buffer.items()):
            if (now - data['arrival_time']).to_sec() > self.buffer_timeout:
                rospy.logwarn(f"Timing out stale trigger {s} (expect: {data['expected_count']}, recv: 0).")
                del self.trigger_buffer[s]

    def _process_batch(self, msgs, topic_list, stamp):
        if not msgs:
            return

        batch_pos, batch_x, valid_topic_list, valid_msgs = [], [], [], []
        
        try:
            for i, msg in enumerate(msgs):
                topic_name = topic_list[i]
                pc_data = ros_numpy.numpify(msg)
                points = np.zeros((len(pc_data), 4), dtype=np.float32)
                points[:, 0], points[:, 1], points[:, 2], points[:, 3] = pc_data['x'], pc_data['y'], pc_data['z'], pc_data['intensity']
                points = points[np.isfinite(points).all(axis=1)]

                if points.shape[0] == 0:
                    rospy.logwarn(f"Empty point cloud on {topic_name} in batch {stamp}. Skipping.")
                    continue

                data = self.processor.process(points)
                batch_pos.append(data['pos'])
                batch_x.append(data['x'])
                valid_topic_list.append(topic_name)
                valid_msgs.append(msg)

            if not batch_pos:
                rospy.logwarn(f"All point clouds in batch for stamp {stamp} were empty or invalid.")
                return 

            pos_tensor = torch.stack(batch_pos, dim=0).to(self.device)
            x_tensor = torch.stack(batch_x, dim=0).to(self.device)
            model_data = {'pos': pos_tensor, 'x': x_tensor.transpose(1, 2)}
            
            with torch.no_grad():
                logits = self.model(model_data)
                pred_indices = torch.argmax(logits, dim=1)
                
                predictions_log = []
                filtered_count = 0
                
                for i in range(pred_indices.shape[0]):
                    class_name = self.class_names[pred_indices[i].item()]
                    topic_name = valid_topic_list[i]
                    
                    predictions_log.append(f"{topic_name}: '{class_name}'")
                    
                    if class_name.lower() != "false":
                        self._publish_individual_filtered_cluster(valid_msgs[i], filtered_count, stamp)
                        filtered_count += 1
                
                # Send completion trigger after all filtered clusters are published
                completion_header = Header()
                completion_header.stamp = stamp
                completion_header.seq = filtered_count
                completion_header.frame_id = "filtered_complete"
                self.completion_trigger_pub.publish(completion_header)
                
                if filtered_count > 0:
                    rospy.logdebug(f"Published {filtered_count} individual filtered clusters from {len(valid_msgs)} total clusters.")
                
                if predictions_log:
                    rospy.logdebug(f"Batch Predictions for stamp {stamp}: {'; '.join(predictions_log)}")

        except Exception as e:
            rospy.logerr(f"Error processing batch for stamp {stamp}: {e}")

    def _publish_individual_filtered_cluster(self, cluster_msg, cluster_index, stamp):
        """
        Republish an individual valid cluster to a filtered topic.
        """
        try:
            filtered_topic = f"{self.filtered_topic_prefix}{cluster_index}"
            
            # Lazily create publisher for this filtered topic.
            if filtered_topic not in self.filtered_publishers:
                self.filtered_publishers[filtered_topic] = rospy.Publisher(
                    filtered_topic, PointCloud2, queue_size=10
                )
                rospy.logdebug(f"Created publisher for {filtered_topic}")

            filtered_msg = PointCloud2()
            filtered_msg.header = cluster_msg.header
            filtered_msg.header.stamp = stamp
            filtered_msg.height = cluster_msg.height
            filtered_msg.width = cluster_msg.width
            filtered_msg.fields = cluster_msg.fields
            filtered_msg.is_bigendian = cluster_msg.is_bigendian
            filtered_msg.point_step = cluster_msg.point_step
            filtered_msg.row_step = cluster_msg.row_step
            filtered_msg.data = cluster_msg.data
            filtered_msg.is_dense = cluster_msg.is_dense
            
            self.filtered_publishers[filtered_topic].publish(filtered_msg)
            
        except Exception as e:
            rospy.logerr(f"Error publishing individual filtered cluster {cluster_index}: {e}")

if __name__ == '__main__':
    try:
        ClassificationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unhandled exception in node initialization: {e}")