import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class VisualizeNode(Node):
    def __init__(self):
        super().__init__('visualize_node')

        # 订阅小球位置的话题
        self.position_subscription = self.create_subscription(
            Float32MultiArray,
            'ball_position',
            self.position_callback,
            0)
        self.position_subscription  # 防止未使用变量的警告

        # 订阅小球速度的话题
        # self.velocity_subscription = self.create_subscription(
        #     Float32MultiArray,
        #     'ball_velocity',
        #     self.velocity_callback,
        #     10)
        # self.velocity_subscription  # 防止未使用变量的警告

        self.ball_position = np.array([0.0, 0.0])  # 初始位置
        self.ball_velocity = np.array([0.0, 0.0])  # 初始速度

        self.fig, self.ax = plt.subplots()
        # 修改robot尺寸：长0.6m，宽0.25m
        self.robot = patches.Rectangle((-0.3, -0.125), 0.6, 0.25, linewidth=1, edgecolor='r', facecolor='none')
        self.ball = patches.Circle((0, 0), 0.1, linewidth=1, edgecolor='b', facecolor='none')
        self.arrow = None
        self.ax.add_patch(self.robot)
        self.ax.add_patch(self.ball)
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal', 'box')
        plt.ion()
        plt.show()

    def position_callback(self, msg):
        self.ball_position = np.array(msg.data[:2])  # 只使用前两个元素作为位置
        print(f"get pos {self.ball_position}")
        self.update_plot()

    def velocity_callback(self, msg):
        self.ball_velocity = np.array(msg.data[:2])  # 只使用前两个元素作为速度
        self.update_plot()

    def update_plot(self):
        # 更新小球的位置
        self.ball.set_center(self.ball_position)
        
        # 更新速度箭头
        if self.arrow:
            self.arrow.remove()
        self.arrow = patches.FancyArrow(self.ball_position[0], self.ball_position[1],
                                        self.ball_velocity[0], self.ball_velocity[1],
                                        head_width=0.05, head_length=0.1, fc='g', ec='g')
        self.ax.add_patch(self.arrow)
        plt.draw()
        plt.pause(0.01)

def main(args=None):
    rclpy.init(args=args)
    visualize_node = VisualizeNode()
    rclpy.spin(visualize_node)
    visualize_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()