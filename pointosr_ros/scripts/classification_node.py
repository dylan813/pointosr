#!/home/cerlab/miniconda3/envs/pointosr/bin/python
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from pointosr_ros.msg import classification_batch, cluster_and_cls
import numpy as np
import torch
import threading
from collections import OrderedDict
import ros_numpy
import time
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
        
        self.batch_publisher = rospy.Publisher(
            '/classified_clusters', classification_batch, queue_size=10
        )
        
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
                rospy.logdebug(f"Trigger for stamp {stamp} received with 0 expected clusters — publishing empty batch.")
                self.buffer_lock.release()
                try:
                    self._process_batch([], [], stamp, 0)
                finally:
                    self.buffer_lock.acquire()
                return

            self.trigger_buffer[stamp] = {'expected_count': expected_count, 'arrival_time': now}

            if stamp in self.frame_buffer:
                if len(self.frame_buffer[stamp]['messages']) == expected_count:
                    self._process_frame_if_complete(stamp)

            self._cleanup_buffers()

    def _process_frame_if_complete(self, stamp):
        rospy.logdebug(f"Frame for stamp {stamp} is complete. Processing.")
        
        batch_data_dict = self.frame_buffer.pop(stamp)['messages']
        expected_count = self.trigger_buffer.pop(stamp)['expected_count']
        
        self.buffer_lock.release()
        
        try:
            topic_list = list(batch_data_dict.keys())
            msgs = [batch_data_dict[tn] for tn in topic_list]
            
            rospy.logdebug(f"Processing batch of {len(msgs)} clusters for stamp {stamp}.")
            self._process_batch(msgs, topic_list, stamp, expected_count)
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

    def _process_batch(self, msgs, topic_list, stamp, expected_count):
        start_time = time.time()
        
        batch_result = classification_batch()
        batch_result.header.stamp = stamp
        batch_result.header.frame_id = "classification_batch"
        batch_result.total_input_clusters = expected_count
        batch_result.total_processed_clusters = 0
        batch_result.human_count = 0
        batch_result.false_count = 0
        batch_result.processing_errors = 0
        batch_result.processing_log = []
        batch_result.classified_clusters = []
        
        if not msgs:
            batch_result.processing_log.append(f"No messages received for stamp {stamp}")
            self._publish_batch_result(batch_result, start_time)
            return

        batch_pos, batch_x, valid_indices, valid_msgs = [], [], [], []
        
        try:
            for i, msg in enumerate(msgs):
                topic_name = topic_list[i]
                
                try:
                    cluster_index = int(topic_name.split('_')[-1])
                except (ValueError, IndexError):
                    cluster_index = i
                
                try:
                    pc_data = ros_numpy.numpify(msg)
                    points = np.zeros((len(pc_data), 4), dtype=np.float32)
                    points[:, 0], points[:, 1], points[:, 2], points[:, 3] = pc_data['x'], pc_data['y'], pc_data['z'], pc_data['intensity']
                    points = points[np.isfinite(points).all(axis=1)]

                    if points.shape[0] == 0:
                        failed_cluster = self._create_failed_cluster_result(msg, cluster_index, "Empty point cloud")
                        batch_result.classified_clusters.append(failed_cluster)
                        batch_result.processing_errors += 1
                        batch_result.processing_log.append(f"Cluster {cluster_index}: Empty point cloud")
                        continue

                    centroid = Point()
                    centroid.x = float(np.mean(points[:, 0]))
                    centroid.y = float(np.mean(points[:, 1]))
                    centroid.z = float(np.mean(points[:, 2]))

                    data = self.processor.process(points)
                    batch_pos.append(data['pos'])
                    batch_x.append(data['x'])
                    valid_indices.append(cluster_index)
                    valid_msgs.append((msg, centroid, len(points)))
                    
                except Exception as e:
                    failed_cluster = self._create_failed_cluster_result(msg, cluster_index, str(e))
                    batch_result.classified_clusters.append(failed_cluster)
                    batch_result.processing_errors += 1
                    batch_result.processing_log.append(f"Cluster {cluster_index}: Processing error - {str(e)}")

            batch_result.total_processed_clusters = len(valid_msgs)

            if not batch_pos:
                batch_result.processing_log.append(f"All {len(msgs)} clusters failed preprocessing")
                self._publish_batch_result(batch_result, start_time)
                return 

            pos_tensor = torch.stack(batch_pos, dim=0).to(self.device)
            x_tensor = torch.stack(batch_x, dim=0).to(self.device)
            model_data = {'pos': pos_tensor, 'x': x_tensor.transpose(1, 2)}
            
            with torch.no_grad():
                logits = self.model(model_data)
                pred_indices = torch.argmax(logits, dim=1)
                confidences = torch.softmax(logits, dim=1)
                
                for i in range(pred_indices.shape[0]):
                    class_idx = pred_indices[i].item()
                    class_name = self.class_names[class_idx]
                    confidence = float(confidences[i, class_idx])
                    cluster_index = valid_indices[i]
                    msg, centroid, cluster_size = valid_msgs[i]
                    
                    cluster_result = cluster_and_cls()
                    cluster_result.header = msg.header
                    cluster_result.header.stamp = stamp
                    cluster_result.pointcloud = msg
                    cluster_result.class_name = class_name
                    cluster_result.confidence = confidence
                    cluster_result.is_human = (class_name.lower() != "false")
                    cluster_result.original_cluster_index = cluster_index
                    cluster_result.processing_success = True
                    cluster_result.error_message = ""
                    cluster_result.centroid = centroid
                    cluster_result.cluster_size = cluster_size
                    
                    batch_result.classified_clusters.append(cluster_result)
                    
                    if cluster_result.is_human:
                        batch_result.human_count += 1
                    else:
                        batch_result.false_count += 1
                        
                    batch_result.processing_log.append(f"Cluster {cluster_index}: '{class_name}' (conf: {confidence:.3f})")

            self._publish_batch_result(batch_result, start_time)

        except Exception as e:
            batch_result.processing_log.append(f"Critical error in batch processing: {str(e)}")
            batch_result.processing_errors += 1
            rospy.logerr(f"Error processing batch for stamp {stamp}: {e}")
            self._publish_batch_result(batch_result, start_time)

    def _create_failed_cluster_result(self, msg, cluster_index, error_message):
        failed_cluster = cluster_and_cls()
        failed_cluster.header = msg.header
        failed_cluster.pointcloud = msg
        failed_cluster.class_name = "processing_failed"
        failed_cluster.confidence = 0.0
        failed_cluster.is_human = False
        failed_cluster.original_cluster_index = cluster_index
        failed_cluster.processing_success = False
        failed_cluster.error_message = error_message
        failed_cluster.centroid = Point()
        failed_cluster.cluster_size = 0
        return failed_cluster
    
    def _publish_batch_result(self, batch_result, start_time):
        try:
            batch_result.processing_time_sec = time.time() - start_time
            batch_result.model_version = "pointnext-s"
            
            self.batch_publisher.publish(batch_result)
            
            total_clusters = batch_result.total_input_clusters
            processed_clusters = batch_result.total_processed_clusters
            human_count = batch_result.human_count
            false_count = batch_result.false_count
            error_count = batch_result.processing_errors
            
            rospy.loginfo(f"Batch {batch_result.header.stamp}: "
                         f"Input={total_clusters}, Processed={processed_clusters}, "
                         f"Human={human_count}, False={false_count}, Errors={error_count}, "
                         f"Time={batch_result.processing_time_sec:.3f}s")
            
            if rospy.get_param('~debug_logging', False):
                for log_entry in batch_result.processing_log:
                    rospy.logdebug(log_entry)
                    
        except Exception as e:
            rospy.logerr(f"Error publishing batch result: {e}")

if __name__ == '__main__':
    try:
        ClassificationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unhandled exception in node initialization: {e}")