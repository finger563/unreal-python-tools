import os
import numpy as np
import cv2
import csv
import json

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.load_camera_config(self.data_path)

    def sanitize_name(self, name):
        return name.replace(' ', '_').lower()

    def load_camera_config(self, folder, camera=None):
        full_config = {}
        config = {}
        if camera:
            camera = self.sanitize_name(camera)
        with open(os.path.join(folder, 'camera_config.csv'), "r") as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                values = row
                cam_name = values[0]
                if camera is not None:
                    print('looking for', camera, 'got', cam_name)
                else:
                    print('loading camera config for', cam_name)
                print('camera:', cam_name)
                print('config:', ', '.join(values))
                # config is : "name", "width", "height", "focalLength", "fov", "nearClipPlane", "farClipPlane"
                full_config[values[0]] = {}
                full_config[values[0]]['name'] = values[0]
                full_config[values[0]]['shape'] = (int(values[2]), int(values[1])) # (height, width)
                full_config[values[0]]['focal length'] = float(values[3])
                full_config[values[0]]['fov'] = float(values[4])
                full_config[values[0]]['near clip plane'] = float(values[5])
                full_config[values[0]]['far clip plane'] = float(values[6])
        self.camera_config = full_config

    def load_raw_image(self, fname, dt, shape):
        w, h, _ = shape
        img = np.fromfile(fname, dt, h*w).reshape(shape)
        return img

    def convert_raw_to_rgb(self, rgb):
        # we don't need to do much except reorder the channels since openCV
        # expects BGR
        rgb = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        return rgb

    def load_convert_rgb(self, img_name, config):
        h, w = config['shape']
        dt = np.dtype(('f4', 4))
        shape = (h, w, 4)
        img = self.load_raw_image(img_name, dt, shape)
        rgb = self.convert_raw_to_rgb(img)
        return rgb

    def convert_raw_to_depth(self, depth, farClipPlane):
        # replace far clip plane to make it easy to cut out. In UE, the clip
        # plane is at infinity, so we just set it here for viz.
        if farClipPlane == float('inf'):
            farClipPlane = 100.0
        # replace far clip plane and any negative depths to only visualize
        # meaningful data
        depth[depth < 0] = 0
        depth[depth >= farClipPlane] = 0
        # so the image isn't all white, convert it to range [0, 1.0]

        # NOTE: for some reason the depth is already in the range 0,1 when
        # capturing data from UE5.2 on MacOS. This is not the case for UE4.26
        # on Windows. So we'll just check the range and convert if necessary.
        _mean, _std = (np.mean(depth), np.std(depth))
        _min, _max = (np.min(depth), np.max(depth))
        print('Depth (min, max, mean, std):', (_min, _max, _mean, _std))
        newMax = _mean + 2 * _std
        newMin = _mean - 2 * _std
        if newMax < _max:
            _max = newMax
        if newMin > _min:
            _min = newMin
        _range = _max-_min
        if _min < 0 and _max > 1 and _range:
            depth -= _min
            depth /= _range
        # convert to 3 channel for visualization
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        return depth

    def convert_raw_to_motion(self, motion):
        # this encoding is copied from the UE5 source code at
        # Engine/Shaders/Private/Common.ush (EncodeVelocityToTexture and DecodeVelocityFromTexture)
        invdiv = 1 / (.499*0.5)
        motion = motion * invdiv - 32767 / 65535 * invdiv
        motion = motion * abs(motion) * .5

        _min, _max = (np.min(motion[..., 0]), np.max(motion[..., 0]))
        _mean, _std = (np.mean(motion[..., 0]), np.std(motion[..., 0]))
        print('Motion x (min, max, mean, std):', (_min, _max, _mean, _std))
        _min, _max = (np.min(motion[..., 1]), np.max(motion[..., 1]))
        _mean, _std = (np.mean(motion[..., 1]), np.std(motion[..., 1]))
        print('Motion y (min, max, mean, std):', (_min, _max, _mean, _std))
        # convert for visualization from x,y vectors to hsv
        # opencv requires image channels = 1, 3, 4
        hsv = np.zeros((motion.shape[0], motion.shape[1], 3), dtype=np.uint8)
        hsv[...,1] = 255
        mag, ang = cv2.cartToPolar(motion[...,0], motion[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        motion = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        # when we stack the images, they all need to be floats
        motion = motion.astype(float) / 255.0
        return motion

    def load_convert_depth_motion(self, img_name, config):
        h, w = config['shape']
        dt = np.dtype(('f4', 4))
        shape = (h, w, 4)
        img = self.load_raw_image(img_name, dt, shape)
        depth = img[..., 0]
        motion = img[..., [1, 2]]
        # convert depth:
        depth = self.convert_raw_to_depth(depth, config['far clip plane'])
        # now convert motion:
        motion = self.convert_raw_to_motion(motion)
        return depth, motion

    def write_text_on_image(self, image, text, origin, centerText=True, textColor=(255,0,0), bgColor=(0,0,0), fontScale=1, thickness=2, bgPadding=5, font=cv2.FONT_HERSHEY_SIMPLEX):
        x, y = origin
        (w, h) = cv2.getTextSize(text, font, fontScale, thickness)[0]
        if centerText:
            textOrigin = (int(x - w/2), int(y + h/2))
            bgBox = ((textOrigin[0] - bgPadding, textOrigin[1] + bgPadding), (int(x + w / 2 + bgPadding), int(y - h / 2 - bgPadding)))
        else:
            textOrigin = origin
            bgBox = (textOrigin, (x+w+bgPadding, y-h-bgPadding))
        cv2.rectangle(image, bgBox[0], bgBox[1], bgColor, cv2.FILLED)
        cv2.putText(image, text, textOrigin, font, fontScale, textColor, thickness, cv2.LINE_AA)

    def get_rgb_image_name(self, camera_name, index):
        return os.path.join(self.data_path, self.sanitize_name(camera_name) + "_" + str(index) + ".raw")

    def get_dmv_image_name(self, camera_name, index):
        return os.path.join(self.data_path, self.sanitize_name(camera_name) + "_depth_motion_" + str(index) + ".raw")

    def display_raw_stack(self, camera_name, index):
        colorImage = self.get_rgb_image_name(camera_name, index)
        previousIndex = index-1
        colorImagePrevious = self.get_rgb_image_name(camera_name, previousIndex)
        # newer format
        depthMotionImage = self.get_dmv_image_name(camera_name, index)

        config = self.camera_config[camera_name]

        c_current = self.load_convert_rgb(colorImage, config)
        try:
            c_previous = self.load_convert_rgb(colorImagePrevious, config)
        except Exception as e:
            print('Warning, could not find previous frame, showing current image instead!')
            c_previous = c_current
            previousIndex = index
        d, m = self.load_convert_depth_motion(depthMotionImage, config)
        dstack = np.vstack((c_current, d))
        mstack = np.vstack((c_previous, m))
        image = np.hstack((dstack, mstack))

        height, width = config['shape']
        # Title for Current
        text = 'Current: ' + str(index)
        self.write_text_on_image(image, text, (0, height), False)
        # Title for Depth
        text = 'Depth: ' + str(index)
        self.write_text_on_image(image, text, (0, height*2), False)
        # Title for Previous
        text = 'Previous: ' + str(previousIndex)
        self.write_text_on_image(image, text, (width, height), False)
        # Title for Motion
        text = 'Motion: ' + str(index)
        self.write_text_on_image(image, text, (width, height*2), False)

        # now show the image
        cv2.imshow(camera_name + ": Color + Depth | Color + Motion", image)
