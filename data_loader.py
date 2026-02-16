import os
import numpy as np
import cv2
import json
from pathlib import Path

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print("Warning: OpenEXR not available. Install with: pip install OpenEXR")

class DataLoader:
    """
    Data loader for Unreal Engine centralized camera capture system.
    
    Supports two formats:
    1. Legacy format: Raw binary files with camera_config.csv
    2. New format: EXR files with JSON metadata per frame
    """
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.format_type = self._detect_format()
        
        if self.format_type == 'legacy':
            self.load_camera_config(self.data_path)
        elif self.format_type == 'centralized':
            self.actors = self._discover_actors()
            print(f"Found {len(self.actors)} actors with cameras")
            for actor_name, cameras in self.actors.items():
                print(f"  {actor_name}: {len(cameras)} camera(s) - {', '.join(cameras)}")
        else:
            raise ValueError(f"Unknown data format in {data_path}")
    
    def _detect_format(self):
        """Detect whether this is legacy or centralized capture format."""
        if (self.data_path / 'camera_config.csv').exists():
            return 'legacy'
        
        # Check for actor/camera directory structure
        subdirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        if subdirs:
            # Check if subdirectories contain camera folders with EXR files
            for actor_dir in subdirs:
                camera_dirs = [d for d in actor_dir.iterdir() if d.is_dir()]
                if camera_dirs:
                    for camera_dir in camera_dirs:
                        exr_files = list(camera_dir.glob('frame_*.exr'))
                        if exr_files:
                            return 'centralized'
        
        return 'unknown'
    
    def _discover_actors(self):
        """Discover actor/camera structure in centralized format."""
        actors = {}
        
        for actor_dir in self.data_path.iterdir():
            if not actor_dir.is_dir():
                continue
            
            actor_name = actor_dir.name
            cameras = []
            
            for camera_dir in actor_dir.iterdir():
                if not camera_dir.is_dir():
                    continue
                
                # Check if this directory has frame data
                exr_files = list(camera_dir.glob('frame_*.exr'))
                if exr_files:
                    cameras.append(camera_dir.name)
            
            if cameras:
                actors[actor_name] = cameras
        
        return actors
    
    def get_camera_path(self, actor_name, camera_name):
        """Get path to camera data directory."""
        if self.format_type != 'centralized':
            raise ValueError("get_camera_path only works with centralized format")
        return self.data_path / actor_name / camera_name
    
    def get_frame_count(self, actor_name, camera_name):
        """Get number of frames captured for a camera."""
        camera_path = self.get_camera_path(actor_name, camera_name)
        exr_files = list(camera_path.glob('frame_*.exr'))
        return len(exr_files)
    
    def load_frame_metadata(self, actor_name, camera_name, frame_number):
        """Load JSON metadata for a specific frame."""
        camera_path = self.get_camera_path(actor_name, camera_name)
        json_path = camera_path / f"frame_{frame_number:07d}.json"
        
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata not found: {json_path}")
        
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def load_frame_exr(self, actor_name, camera_name, frame_number):
        """
        Load EXR frame with all channels (RGB, Depth, Motion).
        
        Returns:
            dict with keys: 'rgb', 'depth', 'motion_x', 'motion_y', 'metadata'
        """
        if not HAS_OPENEXR:
            raise ImportError("OpenEXR library required. Install with: pip install OpenEXR")
        
        camera_path = self.get_camera_path(actor_name, camera_name)
        exr_path = camera_path / f"frame_{frame_number:07d}.exr"
        motion_exr_path = camera_path / f"frame_{frame_number:07d}_motion.exr"
        
        if not exr_path.exists():
            raise FileNotFoundError(f"EXR file not found: {exr_path}")
        
        # Load metadata
        metadata = self.load_frame_metadata(actor_name, camera_name, frame_number)
        
        # Open main EXR file (RGB + Depth)
        exr_file = OpenEXR.InputFile(str(exr_path))
        header = exr_file.header()
        
        # Get image dimensions
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # Get available channels
        channels = header['channels'].keys()
        
        result = {
            'metadata': metadata,
            'width': width,
            'height': height,
            'channels': list(channels)
        }
        
        # Read RGB channels
        if 'R' in channels and 'G' in channels and 'B' in channels:
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            r_str = exr_file.channel('R', pt)
            g_str = exr_file.channel('G', pt)
            b_str = exr_file.channel('B', pt)
            
            r = np.frombuffer(r_str, dtype=np.float32).reshape((height, width))
            g = np.frombuffer(g_str, dtype=np.float32).reshape((height, width))
            b = np.frombuffer(b_str, dtype=np.float32).reshape((height, width))
            
            # Stack as RGB (values in 0-1 range)
            result['rgb'] = np.dstack((r, g, b))
        
        # Read Depth channel (stored in Alpha for now)
        if 'A' in channels:
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            a_str = exr_file.channel('A', pt)
            depth = np.frombuffer(a_str, dtype=np.float32).reshape((height, width))
            result['depth'] = depth
        
        # Read Depth channel if explicitly present
        if 'Depth' in channels:
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            depth_str = exr_file.channel('Depth', pt)
            result['depth'] = np.frombuffer(depth_str, dtype=np.float32).reshape((height, width))
        
        # Read Motion Vector channels from main file
        if 'MotionX' in channels and 'MotionY' in channels:
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            mx_str = exr_file.channel('MotionX', pt)
            my_str = exr_file.channel('MotionY', pt)
            
            result['motion_x'] = np.frombuffer(mx_str, dtype=np.float32).reshape((height, width))
            result['motion_y'] = np.frombuffer(my_str, dtype=np.float32).reshape((height, width))
        
        # Try reading motion from separate _motion.exr file if not in main file
        if 'motion_x' not in result and motion_exr_path.exists():
            try:
                motion_file = OpenEXR.InputFile(str(motion_exr_path))
                motion_header = motion_file.header()
                motion_channels = motion_header['channels'].keys()
                
                # Motion data is stored in R (MotionX) and G (MotionY) channels
                if 'R' in motion_channels and 'G' in motion_channels:
                    pt = Imath.PixelType(Imath.PixelType.FLOAT)
                    mx_str = motion_file.channel('R', pt)
                    my_str = motion_file.channel('G', pt)
                    
                    result['motion_x'] = np.frombuffer(mx_str, dtype=np.float32).reshape((height, width))
                    result['motion_y'] = np.frombuffer(my_str, dtype=np.float32).reshape((height, width))
            except Exception as e:
                print(f"Warning: Failed to load motion data from {motion_exr_path}: {e}")
        
        return result
    
    def convert_rgb_for_display(self, rgb):
        """Convert RGB from float (0-1) to uint8 and BGR for OpenCV."""
        rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    
    def convert_depth_for_display(self, depth, max_depth=None):
        """Convert depth to displayable image with improved visualization."""
        depth_vis = depth.copy()
        
        # Filter out invalid depths (keep track for masking)
        valid_mask = depth_vis > 0
        
        if max_depth is not None:
            valid_mask = valid_mask & (depth_vis < max_depth)
        
        if not valid_mask.any():
            # No valid depth data
            return np.zeros((*depth.shape, 3), dtype=np.uint8)
        
        # Get stats from valid depths only
        valid_depths = depth_vis[valid_mask]
        _min, _max = np.min(valid_depths), np.max(valid_depths)
        _mean, _std = np.mean(valid_depths), np.std(valid_depths)
        
        print(f'Depth stats (valid pixels: {valid_mask.sum()}/{valid_mask.size}):')
        print(f'  Range: {_min:.1f} - {_max:.1f} cm')
        print(f'  Mean: {_mean:.1f} cm, Std: {_std:.1f} cm')
        
        # Use percentile-based ranging for better contrast
        p5, p95 = np.percentile(valid_depths, [5, 95])
        print(f'  Percentiles (5%, 95%): {p5:.1f}, {p95:.1f} cm')
        
        # Normalize using percentiles for better visualization
        range_min, range_max = p5, p95
        depth_normalized = np.clip((depth_vis - range_min) / (range_max - range_min), 0, 1)
        
        # Apply mask - set invalid to 0 (will be black)
        depth_normalized[~valid_mask] = 0
        
        # Apply colormap for better visualization (COLORMAP_TURBO gives good depth perception)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
        
        # Set invalid pixels to black
        depth_colored[~valid_mask] = [0, 0, 0]
        
        return depth_colored
    
    def convert_motion_for_display(self, motion_x, motion_y):
        """Convert motion vectors to HSV visualization."""
        _min, _max = (np.min(motion_x), np.max(motion_x))
        _mean, _std = (np.mean(motion_x), np.std(motion_x))
        print('Motion x (min, max, mean, std):', (_min, _max, _mean, _std))
        
        # filter out any motion vectors which are higher than a threshold
        motion_threshold = 100.0  # pixels per frame, adjust as needed
        valid_mask = (np.abs(motion_x) < motion_threshold) & (np.abs (motion_y) < motion_threshold)
        if not valid_mask.any():
            print("Warning: No valid motion vectors below threshold")
            valid_mask = np.ones_like(motion_x, dtype=bool)  # fallback to all valid

        motion_x = np.where(valid_mask, motion_x, 0)
        motion_y = np.where(valid_mask, motion_y, 0)

        _min, _max = (np.min(motion_y), np.max(motion_y))
        _mean, _std = (np.mean(motion_y), np.std(motion_y))
        print('Motion y (min, max, mean, std):', (_min, _max, _mean, _std))
        
        # Convert to HSV for visualization
        hsv = np.zeros((motion_x.shape[0], motion_x.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255
        
        mag, ang = cv2.cartToPolar(motion_x, motion_y)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        motion_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return motion_bgr
    
    def display_frame(self, actor_name, camera_name, frame_number, previous_frame=None):
        """Display a frame with RGB, Depth, and Motion visualization in separate windows."""
        frame_data = self.load_frame_exr(actor_name, camera_name, frame_number)
        
        if previous_frame is None:
            previous_frame = max(0, frame_number - 1)
        
        results = {}
        
        # Convert and display RGB
        rgb_current = self.convert_rgb_for_display(frame_data['rgb'])
        self.write_text_on_image(rgb_current, f'RGB Frame: {frame_number}', (10, 30), False,
                                textColor=(0, 255, 0), bgColor=(0, 0, 0))
        cv2.imshow(f"{actor_name}/{camera_name} - RGB", rgb_current)
        results['rgb'] = rgb_current
        
        # Convert and display Depth
        if 'depth' in frame_data:
            valid_depth = frame_data['depth'][frame_data['depth'] > 0]
            if len(valid_depth) > 0:
                depth = self.convert_depth_for_display(frame_data['depth'], max_depth=10000)
                
                # Add frame number and depth stats
                depth_stats = f"Depth Frame {frame_number} | Min: {valid_depth.min():.1f}cm  Max: {valid_depth.max():.1f}cm  Mean: {valid_depth.mean():.1f}cm"
                self.write_text_on_image(depth, depth_stats, (10, 30), False,
                                        textColor=(255, 255, 255), bgColor=(0, 0, 0))
                
                cv2.imshow(f"{actor_name}/{camera_name} - Depth", depth)
                results['depth'] = depth
            else:
                print(f"Warning: No valid depth data in frame {frame_number}")
        
        # Convert and display Motion
        if 'motion_x' in frame_data and 'motion_y' in frame_data:
            # Check if there's any actual motion data
            motion_mag = np.sqrt(frame_data['motion_x']**2 + frame_data['motion_y']**2)
            max_motion = motion_mag.max()
            
            if max_motion > 0.001:  # Has actual motion
                motion = self.convert_motion_for_display(frame_data['motion_x'], frame_data['motion_y'])
                motion_stats = f"Motion Frame {frame_number} | Max: {max_motion:.2f}px/f  Mean: {motion_mag.mean():.2f}px/f"
                self.write_text_on_image(motion, motion_stats, (10, 30), False,
                                        textColor=(255, 255, 255), bgColor=(0, 0, 0))
                cv2.imshow(f"{actor_name}/{camera_name} - Motion Vectors", motion)
                results['motion'] = motion
            else:
                print(f"Info: No motion vector data in frame {frame_number} (max magnitude: {max_motion:.6f})")
        
        return results

    # ========================================================================
    # Legacy format support (keep existing methods)
    # ========================================================================

    def sanitize_name(self, name):
        return name.replace(' ', '_').lower()

    def load_camera_config(self, folder, camera=None):
        full_config = {}
        config = {}
        if camera:
            camera = self.sanitize_name(camera)
        with open(os.path.join(folder, 'camera_config.csv'), "r") as file:
            import csv
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
                full_config[cam_name] = {}
                full_config[cam_name]['name'] = cam_name
                full_config[cam_name]['shape'] = (int(values[2]), int(values[1])) # (height, width)
                full_config[cam_name]['focal length'] = float(values[3])
                full_config[cam_name]['fov'] = float(values[4])
                full_config[cam_name]['near clip plane'] = float(values[5])
                full_config[cam_name]['far clip plane'] = float(values[6])
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
            farClipPlane = 5000.0
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
        if _max > 1 and _range:
            depth -= _min
            depth /= _range
        # convert to 3 channel for visualization
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        return depth

    def convert_raw_to_motion(self, motion, threshold=100):
        # filter out any extreme outliers that are likely just noise from the encoding
        motion[np.abs(motion) > threshold] = 0

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

        text = 'Current: ' + str(index)
        self.write_text_on_image(image, text, (0, height), False)

        text = 'Depth: ' + str(index)
        self.write_text_on_image(image, text, (0, height*2), False)

        text = 'Previous: ' + str(previousIndex)
        self.write_text_on_image(image, text, (width, height), False)

        text = 'Motion: ' + str(index)
        self.write_text_on_image(image, text, (width, height*2), False)

        cv2.imshow(camera_name + ": Color + Depth | Color + Motion", image)
