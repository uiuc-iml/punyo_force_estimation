"""
Simple classes to read pressure data from a punyo sensor.

Two classes for reading:
- ROS reader (`PressureReader_ROS`): ROS2 subscriber node that reads pressure data.
    - The problem with this kinda thing is rclpy kinda takes over your python process
      and it's annoying if u want to do anything else with it, like spawn more ROS nodes
      or do threading and things.
- "Python" reader (`PressureReader`):
    - Spawn a ROS pressure reader as a separate process, and communicate with it via
      shared memory.
    - Insulates your program from all the ROS bs and event loop
"""

import rclpy
from rclpy.node import Node

import time
import multiprocessing as mp

from std_msgs.msg import Float32, Int32
from rcl_interfaces.msg import ParameterDescriptor

HISTORY_DEPTH = 5
SENSOR_NAME = "F22EEF8B50583234322E3120FF082009"
# SENSOR_NAME = "AB2BA8A950555733362E3120FF0F2617"
DEFAULT_PRESSURE_TOPIC = "/bubble_{0}/pressure"
DEFAULT_LED_TOPIC = "/bubble_{0}/led"
DEFAULT_DEBUG_MODE = True

class PressureReader:
    def __init__(self):
        self.pressure_shm = mp.Value('d', -1.0)
        self.stop_flag = mp.Value('b', False)
        self.ros_reader = None
        self.proc = None

    def _run(self, pressure_shm, stop_flag, args):
        '''
        Uses the ROS2 Subscriber member function style
        See https://github.com/ros2/examples/blob/rolling/rclpy/topics/minimal_subscriber/examples_rclpy_minimal_subscriber/subscriber_member_function.py
        '''
        rclpy.init(args=args)
        pressure_reader = PressureReader_ROS(pressure_shm)
        while not stop_flag.value:
            rclpy.spin_once(pressure_reader)
        pressure_reader.destroy_node()
        rclpy.shutdown()

    def start(self, args=None):
        self.stop_flag.value = False
        self.proc = mp.Process(target=self._run, args=(self.pressure_shm, self.stop_flag, args))
        self.proc.start()

    def stop(self):
        self.stop_flag.value = True
        self.proc.join()

    def get_pressure(self):
        v = None
        with self.pressure_shm.get_lock():
            v = self.pressure_shm.value
        return v


class PressureReader_ROS(Node):
    def __init__(self, shm=None, sensor_name=SENSOR_NAME):
        super().__init__('read_pressure')

        # Topics are parameters, specified in the launch file or a yaml
        self.declare_parameter('pressure_topic_parameter', DEFAULT_PRESSURE_TOPIC.format(sensor_name),
                               ParameterDescriptor(description='Topic to receive pressure data'))
        self.declare_parameter('led_topic_parameter', DEFAULT_LED_TOPIC.format(sensor_name),
                               ParameterDescriptor(description='Topic to send LED color'))
        self.declare_parameter('debug_parameter', DEFAULT_DEBUG_MODE,
                               ParameterDescriptor(description='Debug mode'))

        # topics
        pressure_topic = self.get_parameter('pressure_topic_parameter').get_parameter_value().string_value
        led_topic = self.get_parameter('led_topic_parameter').get_parameter_value().string_value

        # misc params
        self._debug = self.get_parameter('debug_parameter').get_parameter_value().bool_value

        # Subscribers
        # best_effort_qos = rclpy.qos.QoSProfile(
        #     history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=HISTORY_DEPTH,
        #     reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT
        # )
        best_effort_qos = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                               history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=5)
        self._pressure_sub = self.create_subscription(Float32, pressure_topic, self.pressure_callback, best_effort_qos)
        print(f"INFO: subscribe topic {pressure_topic}")

        # Publishers
        self._led_pub = self.create_publisher(Int32, led_topic, HISTORY_DEPTH)
        msg = Int32()
        msg.data = 0xffffff
        self._led_pub.publish(msg)

        # Pressure shared memory
        self._pressure_shm = shm

        print("INFO: ROS node initialized")

    def pressure_callback(self, msg):
        # print("pressure reading:", msg.data)

        pressure = msg.data
        if self._pressure_shm is not None:
            with self._pressure_shm.get_lock():
                self._pressure_shm.value = pressure

def raw_subscribe(args=None):
    rclpy.init(args=args)
    minimal_subscriber = PressureReader_ROS()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

def main(args=None):
    reader = PressureReader()
    reader.start()
    time.sleep(1)
    for i in range(10):
        print(reader.get_pressure())
        time.sleep(1)
    reader.stop()

if __name__ == '__main__':
    # raw_subscribe()
    main()
