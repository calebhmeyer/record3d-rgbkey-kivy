import numpy as np
from record3d import Record3DStream
import cv2
from kivy.app import App
from kivy.uix.slider import Slider
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
import threading
from threading import Event
import time


def darken_top_pixels(img, toggle_button_state, percentile=20):
    """
    Sets the brightness of the top X percentile of pixels in the image to 0.
    
    Args:
    - img: The image as a numpy array.
    - toggle_button_state: The state of the toggle button.
    - percentile: The percentile of pixels to darken (default is 20).
    
    Returns:
    - The image with the top X percentile of pixels darkened.
    """
    # Check if the toggle_button_state is 'down'
    if toggle_button_state == 'down':
        # Convert image to grayscale to get brightness
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = img

        # Remove 0 brightness pixels
        non_zero_pixels = gray[gray > 0]

        if len(non_zero_pixels) == 0:
            return np.zeros_like(img)

        # Find the brightness value that represents the 20th percentile
        thresh = np.percentile(non_zero_pixels, percentile)

        # Create a mask of all pixels that are above this value
        mask = gray > thresh

        # Darken all pixels above the threshold by setting them to 0
        img[mask] = 0

    return img


class MyApp(App):
    max_depth = 5.0
    offset = 0
    clip_threshold = 10.0
    darken_percentile = 20.0 # default darken percentile


    def on_start(self):
        self.app.connect_to_device(dev_idx=0) # Add this line
        thread = threading.Thread(target=self.app.start_processing_stream)
        thread.daemon = True
        thread.start()

    def build(self):
        layout = BoxLayout(orientation='vertical')

        self.label1 = Label(text="Max Depth Cam 1: 5.00", size_hint=(1, .2), font_size='20sp')
        self.slider1 = Slider(orientation='horizontal', min=0, max=10, value=5, size_hint=(1, .2))
        self.slider1.bind(value=self.on_value_change_1)

        self.label2 = Label(text="Offset Cam 1: 1.00", size_hint=(1, .2), font_size='20sp')
        self.slider2 = Slider(orientation='horizontal', min=-2, max=2, value=0, size_hint=(1, .2))
        self.slider2.bind(value=self.on_value_change_2)

        self.label3 = Label(text="Clip Threshold Cam 1: 10.00", size_hint=(1, .2), font_size='20sp')
        self.slider3 = Slider(orientation='horizontal', min=0, max=20, value=10, size_hint=(1, .2))
        self.slider3.bind(value=self.on_value_change_3)

        # Create fourth slider and label for darken percentile
        self.label4 = Label(text="Darken Percentile: 20.00", size_hint=(1, .2), font_size='20sp')
        self.slider4 = Slider(orientation='horizontal', min=0, max=100, value=20, size_hint=(1, .2))
        self.slider4.bind(value=self.on_value_change_4)

        layout.add_widget(self.label1)
        layout.add_widget(self.slider1)
        layout.add_widget(self.label2)
        layout.add_widget(self.slider2)
        layout.add_widget(self.label3)
        layout.add_widget(self.slider3)
        layout.add_widget(self.label4)
        layout.add_widget(self.slider4)

        self.toggle_button = ToggleButton(text='Toggle Darken Filter', size_hint=(1, .2))
        layout.add_widget(self.toggle_button)

        self.key_button = ToggleButton(text='Toggle Key', size_hint=(1, .2))
        layout.add_widget(self.key_button)
        
        return layout

    def on_value_change_1(self, instance, value):
        self.max_depth = value
        self.label1.text = "Max Depth: %.2f" % value

    def on_value_change_2(self, instance, value):
        self.offset = value
        self.label2.text = "Offset: %.2f" % value

    def on_value_change_3(self, instance, value):
        self.clip_threshold = value
        self.label3.text = "Clip Threshold: %.2f" % value

    def on_value_change_4(self, instance, value):
        self.darken_percentile = value
        self.label4.text = "Darken Percentile: %.2f" % value





class DemoApp:
    def __init__(self, my_app):
        self.my_app = my_app
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1

    

    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print('Stream stopped')

    def connect_to_device(self, dev_idx):
        print('Searching for devices')
        devs = Record3DStream.get_connected_devices()
        print('{} device(s) found'.format(len(devs)))
        for dev in devs:
            print('\tID: {}\n\tUDID: {}\n'.format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError('Cannot connect to device #{}, try different index.'
                               .format(dev_idx))

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array([[coeffs.fx,         0, coeffs.tx],
                         [        0, coeffs.fy, coeffs.ty],
                         [        0,         0,         1]])

    def start_processing_stream(self):
        while True:
            self.event.wait()

            depth = self.session.get_depth_frame()#

            rgb = self.session.get_rgb_frame()

            rgb_resized = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
            rgb_resized = cv2.flip(rgb_resized, 1)  # Flip horizontally

            intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
            camera_pose = self.session.get_camera_pose()

            print('Running')

            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)

            rgb_resized = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR)

            toggle_button_state = self.my_app.toggle_button.state
            key_button_state = self.my_app.key_button.state

            #Respoinsible for clip threshhold
            depth[depth > self.my_app.clip_threshold] = 0

            darkendepth = darken_top_pixels(depth.copy(), toggle_button_state, percentile=self.my_app.darken_percentile)

            if toggle_button_state == 'down':
                depth = darkendepth

            #depth = cv2.GaussianBlur(depth.astype(np.float32), (5, 5), 0)
            depth = depth - self.my_app.offset
            depth_mask = depth > 0.05 #boolean 
            if key_button_state == 'down':
                rgb_resized[~depth_mask] = [0, 255, 0]
            cv2.imshow('RGB Camera', rgb_resized)
            #cv2.imshow('Depth Camera', depth / self.my_app.max_depth)
            #cv2.imshow('depth mask', depth_mask.astype(np.uint8) * 255)
            cv2.waitKey(1)
            time.sleep(.01)

            self.event.clear()





if __name__ == '__main__':
    my_app_instance = MyApp()
    app = DemoApp(my_app_instance)
    my_app_instance.app = app
    my_app_instance.run()
