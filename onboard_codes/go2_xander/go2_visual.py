import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import Image, CameraInfo

import os
import os.path as osp
import json
import time
from collections import OrderedDict
import numpy as np
import cv2
import pickle


class FisheyeCameraNode(Node):
    def __init__(self,
            cfg:dict,
            detection_info_topic = "/ball_position",
            camera_extrinsics_topic = '/camera_extrinsics',
    ):
        super().__init__("fisheye_camera_node")
        self.cfg = cfg
        
        self.detection_info_topic = detection_info_topic
        self.camera_extrinsics_topic = camera_extrinsics_topic
        
        self.start_ros_handlers()
        self.start_pipeline()
        
        # 初始化外参 (默认单位矩阵)
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        
        self.get_logger().info('FisheyeCameraNode initialized.')
        
    def start_ros_handlers(self):
        # 发布相机和世界坐标的话题
        self.detection_info_pub = self.create_publisher(Float32MultiArray, 
                                                        self.detection_info_topic,
                                                        10)
        # 订阅外参的节点, 例如: 外参消息定义为 R 和 t
        self.camera_extrinsics_sub = self.create_subscription(String,
                                                              self.camera_extrinsics_topic,
                                                              self.extrinsic_callback,
                                                              10)
    
    def start_pipeline(self):
        self.pipeline = "/dev/video0"
        self.capture = cv2.VideoCapture(self.pipeline)
        if not self.capture.isOpened():
            self.get_logger().error("Failed to open camera")
        else:
            self.get_logger().info("Camera opened successfully")

        # 读取鱼眼相机的内参矩阵 K 和畸变参数
        with open(self.cfg.calibration_params, 'rb') as f:
            calibration_params = pickle.load(f)
        self.K = calibration_params['K']
        self.D = calibration_params['D']
        self.DIM = calibration_params['DIM']

    def undistort(self, img, K, D, DIM, scale=0.6):
        """自定义去畸变函数."""
        # TODO: 看起来没用到?
        if img is None:
            return None
        dim1 = img.shape[:2][::-1]
        if dim1[0] / dim1[1] != DIM[0] / DIM[1]:
            img = cv2.resize(img, DIM, interpolation=cv2.INTER_AREA)

        Knew = K.copy()
        if scale:
            Knew[(0, 1), (0, 1)] = scale * Knew[(0, 1), (0, 1)]

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img
        
    def extrinsic_callback(self, msg):
        """接收外参消息，更新旋转矩阵及平移向量"""
        try:
            # 解析外参，假设消息格式为字符串化的数组，这里为了简单直接使用eval
            extrinsic_data = eval(msg.data)
            self.R = np.array(extrinsic_data['R'])  # 旋转矩阵
            self.t = np.array(extrinsic_data['t'])  # 平移向量
            self.get_logger().info('Extrinsics updated: R and t received.')
        except (KeyError, SyntaxError) as e:
            self.get_logger().error(f"Failed to parse extrinsics: {e}")
    
    def start_main_loop_timer(self, duration):
        """ Start the main loop timer when using the timer mode """
        self.create_timer(
            duration,
            self.main_loop,
        )
    
    def main_loop(self):
        retval, image = self.capture.read()
        if not retval:
            self.get_logger().error("Failed to read from camera.")
            return

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = (40, 40, 40)
        upper_green = (80, 255, 255)
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        known_diameter = 0.18  # 物体的已知直径（米）

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            cx, cy = x + w // 2, y + h // 2
            pixel_diameter = w

            if pixel_diameter > 0:
                # 计算 Z 轴上的距离
                focal_length = self.K[0, 0]
                distance = (known_diameter * focal_length) / pixel_diameter

                # 归一化坐标
                x_dis = (cx - image.shape[1] / 2)
                y_dis = (cy - image.shape[0] / 2)
                x_normal = x_dis / focal_length
                y_normal = y_dis / focal_length

                #球心与图像中心距离r
                r = np.sqrt(x_dis**2 + y_dis**2)

                # theta = r / f
                angle_theta = r / focal_length
                angle_theta = np.clip(angle_theta,0,np.pi/2)
                
                angle_phi = np.arctan2(y_normal, x_normal)

                # 转换为相机坐标系下的3D坐标
                X_cam = distance * np.sin(angle_theta) * np.cos(angle_phi)
                Y_cam = distance * np.sin(angle_theta) * np.sin(angle_phi)
                Z_cam = distance * np.cos(angle_theta)

                # 将相机坐标转换为世界坐标
                camera_coords = np.array([X_cam, Y_cam, Z_cam]).reshape(3, 1)
                world_coords = self.R @ camera_coords + self.t

                # 发布位置、距离以及世界坐标信息
                output_message = (
                    f"Camera Coordinates (X, Y, Z): ({X_cam:.2f}, {Y_cam:.2f}, {Z_cam:.2f}), "
                    f"World Coordinates (X_w, Y_w, Z_w): ({world_coords[0, 0]:.2f}, {world_coords[1, 0]:.2f}, {world_coords[2, 0]:.2f}), "
                    f"Distance: {distance:.2f}m, Angles (Theta, Phi): ({np.degrees(angle_theta):.2f}°, {np.degrees(angle_phi):.2f}°)"
                )

                self.detection_info_pub.publish(Float32MultiArray(data=[-5*Y_cam,-5*X_cam,0.0]))
                self.get_logger().info(output_message)

                # 在图像上绘制
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)
            else:
                # TODO: 是否应该给上一时刻的检测值?
                self.get_logger().info("Object too small or too distant")
                self.detection_info_pub.publish(Float32MultiArray(data=[0.2,0.0,0.0]))
        else:
            # TODO: 是否应该给上一时刻的检测值?
            self.get_logger().info("No object detected")
            self.detection_info_pub.publish(Float32MultiArray(data=[0.2,0.0,0.0]))

        # cv2.imshow("Camera Output", image)
        # cv2.waitKey(1)


def main(args):
    rclpy.init()
    fisheye_camera_node = FisheyeCameraNode(cfg=args,)
    
    duration = args.duration  # in sec
    fisheye_camera_node.get_logger().info("Ball position send duration: {:.2f} sec".format(duration))

    if args.loop_mode == "while":
        rclpy.spin_once(fisheye_camera_node, timeout_sec = 0.)
        while rclpy.ok():
            main_loop_time = time.monotonic()
            fisheye_camera_node.main_loop()
            rclpy.spin_once(fisheye_camera_node, timeout_sec= 0.)
            time.sleep(max(0, duration - (time.monotonic() - main_loop_time)))
    elif args.loop_mode == "timer":
        fisheye_camera_node.start_main_loop_timer(duration)
        rclpy.spin(fisheye_camera_node)

    fisheye_camera_node.capture.release()
    cv2.destroyAllWindows()
    fisheye_camera_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # TODO: add more arguments if needed
    parser.add_argument("--loop_mode", type = str, default = "timer",
        choices= ["while", "timer"],
        help= "Select which mode to run the main policy control iteration",
    )
    parser.add_argument("--device", type = str, default = "cpu", help = "The device to run the model")
    parser.add_argument("--duration", type = float, default = 0.1, help = "The duration of the main loop in sec")
    parser.add_argument("--calibration_params", type = str, default = "./calibration_params.pickle", help = "The path to the calibration parameters")
    
    
    args = parser.parse_args()
    main(args)