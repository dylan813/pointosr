import rospy
import time

def count_cluster_topics():
    topics = rospy.get_published_topics()
    cluster_count = sum(1 for topic, _ in topics if topic.startswith('/cluster_'))
    return cluster_count

def main():
    rospy.init_node('count_clusters', anonymous=True)
    buffer_size = 10  # Number of frames to buffer
    buffer = []

    while not rospy.is_shutdown():
        # Capture the current frame's data
        frame_data = {
            'timestamp': rospy.get_time(),
            'cluster_count': count_cluster_topics()
        }
        buffer.append(frame_data)

        if len(buffer) > buffer_size:
            buffer.pop(0)  # Remove the oldest frame data

        # Print the cluster count for the current frame
        print(f"Timestamp: {frame_data['timestamp']}, Cluster Count: {frame_data['cluster_count']}")

        time.sleep(1)  # Adjust the sleep time as needed

if __name__ == "__main__":
    main()