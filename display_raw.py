import cv2
from data_loader import DataLoader
import time
import numpy as np
import argparse

def display_single_camera(dataLoader, actor_name, camera_name, start_frame=0):
    """Display frames from a single camera."""
    fps = 30
    index = start_frame
    playing = False

    print(f"\nDisplaying: {actor_name}/{camera_name}")
    print("Controls:")
    print("  Space: Play/Pause")
    print("  N: Next frame")
    print("  P: Previous frame")
    print("  Q or ESC: Quit")
    print()

    while True:
        try:
            if dataLoader.format_type == 'centralized':
                dataLoader.display_frame(actor_name, camera_name, index)
            else:
                # Legacy format
                dataLoader.display_raw_stack(camera_name, index)
        except Exception as e:
            print(f"Exception at frame {index}: {e}")
            break

        while True:
            key = cv2.waitKey(int(1000.0/fps))
            if key == 27 or key == ord('q'):  # ESC or Q
                return
            elif key == 32:  # Space - play/pause
                playing = not playing
                break
            elif key == ord('p'):  # P - previous
                index = max(index - 1, 0)
                break
            elif key == ord('n'):  # N - next
                index += 1
                break
            elif playing:
                index += 1
                break

def display_all_cameras(dataLoader, actor_name, start_frame=0):
    """Display frames from all cameras of an actor in a grid."""
    if dataLoader.format_type != 'centralized':
        print("Error: Multi-camera display only works with centralized format")
        return
    
    if actor_name not in dataLoader.actors:
        print(f"Error: Actor '{actor_name}' not found")
        print(f"Available actors: {', '.join(dataLoader.actors.keys())}")
        return
    
    cameras = dataLoader.actors[actor_name]
    print(f"\nDisplaying all {len(cameras)} cameras for actor: {actor_name}")
    print(f"Cameras: {', '.join(cameras)}")
    print("\nControls:")
    print("  Space: Play/Pause")
    print("  N: Next frame")
    print("  P: Previous frame")
    print("  Q or ESC: Quit")
    print()
    
    fps = 30
    index = start_frame
    playing = False
    
    while True:
        try:
            # Load frames from all cameras
            frames = []
            for camera_name in cameras:
                try:
                    frame_data = dataLoader.load_frame_exr(actor_name, camera_name, index)
                    rgb = dataLoader.convert_rgb_for_display(frame_data['rgb'])
                    
                    # Add camera name label
                    dataLoader.write_text_on_image(
                        rgb, f"{camera_name} - Frame {index}", 
                        (10, 30), False, 
                        textColor=(0, 255, 0), bgColor=(0, 0, 0)
                    )
                    
                    frames.append(rgb)
                except Exception as e:
                    print(f"Warning: Could not load {camera_name} frame {index}: {e}")
                    # Create blank frame
                    if frames:
                        blank = np.zeros_like(frames[0])
                        dataLoader.write_text_on_image(
                            blank, f"{camera_name}: Error", 
                            (10, 30), False,
                            textColor=(0, 0, 255), bgColor=(0, 0, 0)
                        )
                        frames.append(blank)
            
            if not frames:
                print(f"No frames loaded for index {index}")
                break
            
            # Arrange frames in a grid
            num_cameras = len(frames)
            
            if num_cameras == 1:
                grid = frames[0]
            elif num_cameras == 2:
                # 1x2 horizontal
                grid = np.hstack(frames)
            elif num_cameras == 3:
                # 1x3 horizontal or 2x2 with one empty
                grid = np.hstack(frames)
            elif num_cameras == 4:
                # 2x2 grid
                top_row = np.hstack(frames[:2])
                bottom_row = np.hstack(frames[2:4])
                grid = np.vstack([top_row, bottom_row])
            else:
                # Calculate grid dimensions
                cols = int(np.ceil(np.sqrt(num_cameras)))
                rows = int(np.ceil(num_cameras / cols))
                
                # Pad frames to fill grid
                while len(frames) < rows * cols:
                    frames.append(np.zeros_like(frames[0]))
                
                # Build grid row by row
                grid_rows = []
                for r in range(rows):
                    row_frames = frames[r * cols:(r + 1) * cols]
                    grid_rows.append(np.hstack(row_frames))
                
                grid = np.vstack(grid_rows)
            
            # Display
            cv2.imshow(f"{actor_name} - All Cameras", grid)
            
        except Exception as e:
            print(f"Exception at frame {index}: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # Handle keyboard input
        while True:
            key = cv2.waitKey(int(1000.0/fps))
            if key == 27 or key == ord('q'):  # ESC or Q
                cv2.destroyAllWindows()
                return
            elif key == 32:  # Space - play/pause
                playing = not playing
                print("Playing" if playing else "Paused")
                break
            elif key == ord('p'):  # P - previous
                index = max(index - 1, 0)
                break
            elif key == ord('n'):  # N - next
                index += 1
                break
            elif playing:
                index += 1
                break

def main():
    parser = argparse.ArgumentParser(
        description='Display captured camera data from Unreal Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Legacy format (single camera)
  python display_raw.py data/capture camera_name
  
  # Centralized format (single camera)
  python display_raw.py data/capture --actor Gen3_BP_C --camera RightEyeCamera
  
  # Centralized format (all cameras on actor)
  python display_raw.py data/capture --actor Gen3_BP_C --all-cameras
  
  # Start from specific frame
  python display_raw.py data/capture --actor Gen3_BP_C --all-cameras --start-frame 100
        """
    )
    
    parser.add_argument('folder_path', help='Path to captured data folder')
    parser.add_argument('camera_name', nargs='?', help='Camera name (legacy format)')
    parser.add_argument('--actor', '-a', help='Actor name (centralized format)')
    parser.add_argument('--camera', '-c', help='Camera name (centralized format)')
    parser.add_argument('--all-cameras', action='store_true', 
                       help='Display all cameras for the actor in a grid')
    parser.add_argument('--start-frame', '-s', type=int, default=0,
                       help='Starting frame number (default: 0)')
    
    args = parser.parse_args()
    
    # Load data
    dataLoader = DataLoader(args.folder_path)
    
    print(f"Detected format: {dataLoader.format_type}")
    
    if dataLoader.format_type == 'legacy':
        # Legacy format - requires camera name
        if not args.camera_name:
            print("Error: Camera name required for legacy format")
            print("Usage: python display_raw.py <folder_path> <camera_name>")
            return
        
        display_single_camera(dataLoader, None, args.camera_name, args.start_frame)
    
    elif dataLoader.format_type == 'centralized':
        # Centralized format
        if not args.actor:
            # Use first actor if not specified
            args.actor = list(dataLoader.actors.keys())[0]
            print(f"No actor specified, using first actor: {args.actor}")
        
        if args.actor not in dataLoader.actors:
            print(f"Error: Actor '{args.actor}' not found")
            print(f"Available actors: {', '.join(dataLoader.actors.keys())}")
            return
        
        if args.all_cameras:
            # Display all cameras in grid
            display_all_cameras(dataLoader, args.actor, args.start_frame)
        else:
            # Single camera
            if not args.camera:
                # Use first camera
                args.camera = dataLoader.actors[args.actor][0]
                print(f"No camera specified, using first camera: {args.camera}")
            
            if args.camera not in dataLoader.actors[args.actor]:
                print(f"Error: Camera '{args.camera}' not found on actor '{args.actor}'")
                print(f"Available cameras: {', '.join(dataLoader.actors[args.actor])}")
                return
            
            display_single_camera(dataLoader, args.actor, args.camera, args.start_frame)
    
    else:
        print(f"Error: Unknown data format")
        return
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

