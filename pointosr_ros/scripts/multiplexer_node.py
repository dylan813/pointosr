#!/usr/bin/env python3
import rospy
import threading
from functools import partial
from sensor_msgs.msg import PointCloud2
from pointosr_ros.msg import PointCloud2Array

class FrameBasedMultiplexer:
    """
    A robust, frame-based multiplexer for dynamically changing topics.
    It discovers topics, creates individual subscribers, buffers messages
    by their exact timestamp (frame), and publishes perfectly synchronized
    batches of data.
    """
    def __init__(self):
        rospy.init_node('pointcloud_frame_multiplexer_node')

        # --- Parameters ---
        self.input_topic_prefix = rospy.get_param('~input_topic_prefix', '/cluster_')
        self.output_topic = rospy.get_param('~output_topic', '/clusters_aggregated')
        publish_frequency = rospy.get_param('~publish_frequency', 20.0)
        discovery_frequency = rospy.get_param('~discovery_frequency', 0.5) # Check for new topics every 500ms
        self.max_buffer_age = rospy.Duration(rospy.get_param('~max_buffer_age_secs', 2.0))

        # --- Data Structures & Synchronization ---
        self.lock = threading.Lock()
        self.active_topics = set()
        self.subscribers = {}
        self.buffers = {} # Shape: {topic_name: {timestamp: msg}}

        # --- Publisher ---
        self.agg_publisher = rospy.Publisher(self.output_topic, PointCloud2Array, queue_size=5)

        rospy.loginfo("Frame-based multiplexer initialized.")
        rospy.loginfo(f"Discovering topics with prefix '{self.input_topic_prefix}'.")
        rospy.loginfo(f"Publishing aggregated frames to '{self.output_topic}' at {publish_frequency} Hz.")

        # --- Timers ---
        self.discovery_timer = rospy.Timer(
            rospy.Duration(1.0 / discovery_frequency), self._discover_topics)
        
        self.publish_timer = rospy.Timer(
            rospy.Duration(1.0 / publish_frequency), self._aggregate_and_publish)

    def _message_callback(self, msg, topic_name):
        """
        Callback for each individual subscriber. Places the message into the
        correct buffer, keyed by its timestamp.
        """
        with self.lock:
            # Ensure the buffer for this topic exists
            if topic_name in self.buffers:
                self.buffers[topic_name][msg.header.stamp] = msg

    def _discover_topics(self, event=None):
        """
        Periodically checks for new or removed topics and surgically
        updates the subscribers without reconfiguring the entire system.
        """
        try:
            # Find all topics matching the prefix
            published_topics = [
                topic for topic, msg_type in rospy.get_published_topics()
                if topic.startswith(self.input_topic_prefix) and msg_type == 'sensor_msgs/PointCloud2'
            ]
            found_topics = set(published_topics)

            with self.lock:
                new_topics = found_topics - self.active_topics
                dead_topics = self.active_topics - found_topics

                # Add subscribers for new topics
                for topic in new_topics:
                    rospy.loginfo(f"Discovered new topic: {topic}")
                    self.buffers[topic] = {}
                    # Use functools.partial to pass the topic name to the callback
                    callback = partial(self._message_callback, topic_name=topic)
                    self.subscribers[topic] = rospy.Subscriber(topic, PointCloud2, callback, queue_size=20)
                self.active_topics.update(new_topics)

                # Remove subscribers for dead topics
                for topic in dead_topics:
                    rospy.loginfo(f"Topic disappeared: {topic}")
                    if topic in self.subscribers:
                        self.subscribers[topic].unregister()
                        del self.subscribers[topic]
                    if topic in self.buffers:
                        del self.buffers[topic]
                self.active_topics.difference_update(dead_topics)

        except Exception as e:
            rospy.logerr(f"Error in topic discovery: {e}", exc_info=True)

    def _aggregate_and_publish(self, event=None):
        """
        Finds frames (timestamps) that are present in ALL active buffers,
        packages them, publishes them, and cleans them from the buffers.
        """
        with self.lock:
            if not self.active_topics:
                return

            # Find timestamps present in the first buffer as a starting point
            # This avoids checking against an empty set if a new buffer was just added
            first_topic = next(iter(self.active_topics))
            complete_frames = set(self.buffers[first_topic].keys())

            # Find the intersection of keys (timestamps) across all buffers
            for topic in self.active_topics:
                complete_frames.intersection_update(self.buffers[topic].keys())
            
            # --- Publish complete frames ---
            for frame_stamp in sorted(list(complete_frames)):
                aggregated_msg = PointCloud2Array()
                aggregated_msg.header.stamp = frame_stamp
                # Frame ID should be consistent, get it from the first message
                aggregated_msg.header.frame_id = self.buffers[first_topic][frame_stamp].header.frame_id
                
                # Collect all messages for this frame
                for topic in self.active_topics:
                    msg = self.buffers[topic].pop(frame_stamp) # Pop removes the item
                    aggregated_msg.clouds.append(msg)
                    aggregated_msg.topic_names.append(topic)
                
                rospy.loginfo_throttle(1.0, f"Publishing complete frame {frame_stamp.to_sec()} with {len(aggregated_msg.clouds)} clouds.")
                self.agg_publisher.publish(aggregated_msg)

            # --- Clean up old, incomplete frames to prevent memory leaks ---
            now = rospy.Time.now()
            for topic in self.active_topics:
                stamps_to_delete = [
                    stamp for stamp in self.buffers[topic]
                    if now - stamp > self.max_buffer_age
                ]
                for stamp in stamps_to_delete:
                    rospy.logwarn_throttle(10.0, f"Purging old, incomplete frame {stamp.to_sec()} from topic {topic}.")
                    del self.buffers[topic][stamp]


if __name__ == '__main__':
    try:
        FrameBasedMultiplexer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unhandled exception in multiplexer node: {e}", exc_info=True) 