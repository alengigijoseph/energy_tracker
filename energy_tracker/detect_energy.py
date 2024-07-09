import rclpy
from rclpy.node import Node
from sensor_msgs.msg        import Image
from geometry_msgs.msg      import Point
from cv_bridge              import CvBridge, CvBridgeError
import energy_tracker.energy_tracker.process_image as proc
from std_msgs.msg import String

class DetectEnergy(Node):

    def __init__(self):
        super().__init__('detect_energy')

        self.get_logger().info('Looking for the energy...')
        self.image_sub = self.create_subscription(Image,"/image_in",self.callback,rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.energy_type = self.create_subscription(String,"/energy_type",self.callbackk)

        self.image_out_pub = self.create_publisher(Image, "/image_out", 1)
        self.image_tuning_pub = self.create_publisher(Image, "/image_tuning", 1)
        self.ball_pub  = self.create_publisher(Point,"/detected_energy",1)

        self.declare_parameter('tuning_mode', False)

        self.declare_parameter("x_min",0)
        self.declare_parameter("x_max",100)
        self.declare_parameter("y_min",0)
        self.declare_parameter("y_max",100)
        self.declare_parameter("h_min",20) #0
        self.declare_parameter("h_max",30) #180
        self.declare_parameter("s_min",50) #0
        self.declare_parameter("s_max",255)
        self.declare_parameter("v_min",50) #0
        self.declare_parameter("v_max",255)
        self.declare_parameter("sz_min",0)
        self.declare_parameter("sz_max",100)
        

        self.tuning_mode = self.get_parameter('tuning_mode').get_parameter_value().bool_value

        self.bridge = CvBridge()

        if(self.tuning_mode):
            proc.create_tuning_window(self.tuning_params)

    def callbackk(self,msg):
        if msg.data == "blue":
            self.tuning_params = {
            'x_min': 0,
            'x_max':100,
            'y_min': 0,
            'y_max': 100,
            'h_min': 100,
            'h_max': 130,
            's_min': 50,
            's_max': 255,
            'v_min': 50,
            'v_max': 255,
            'sz_min': 0,
            'sz_max': 100  }
        elif msg.data == "yellow":
            self.tuning_params = {
           'x_min': 0,
            'x_max':100,
            'y_min': 0,
            'y_max': 100,
            'h_min': 20,
            'h_max': 30,
            's_min': 50,
            's_max': 255,
            'v_min': 50,
            'v_max': 255,
            'sz_min': 0,
            'sz_max': 100   }
        elif msg.data == "green":
            self.tuning_params = {
            'x_min': 0,
            'x_max':100,
            'y_min': 0,
            'y_max': 100,
            'h_min': 60,
            'h_max': 85,
            's_min': 50,
            's_max': 255,
            'v_min': 50,
            'v_max': 255,
            'sz_min': 0,
            'sz_max': 100 }



    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        try:
            if (self.tuning_mode):
                self.tuning_params = proc.get_tuning_params()

            keypoints_norm, out_image, tuning_image = proc.find_energy(cv_image, self.tuning_params)

            img_to_pub = self.bridge.cv2_to_imgmsg(out_image, "bgr8")
            img_to_pub.header = data.header
            self.image_out_pub.publish(img_to_pub)

            img_to_pub = self.bridge.cv2_to_imgmsg(tuning_image, "bgr8")
            img_to_pub.header = data.header
            self.image_tuning_pub.publish(img_to_pub)

            point_out = Point()

            # Keep the biggest point
            # They are already converted to normalised coordinates
            for i, kp in enumerate(keypoints_norm):
                x = kp.pt[0]
                y = kp.pt[1]
                s = kp.size

                self.get_logger().info(f"Pt {i}: ({x},{y},{s})")

                if (s > point_out.z):                    
                    point_out.x = x
                    point_out.y = y
                    point_out.z = s

            if (point_out.z > 0):
                self.ball_pub.publish(point_out) 
        except CvBridgeError as e:
            print(e)  


def main(args=None):

    rclpy.init(args=args)

    detect_ball = DetectEnergy()
    while rclpy.ok():
        rclpy.spin_once(detect_ball)
        proc.wait_on_gui()

    detect_ball.destroy_node()
    rclpy.shutdown()

