"""
Example script for displaying data from the centralized camera capture system.

Usage:
    python display_centralized.py <data_path> [actor_name] [camera_name] [start_frame]
    
Example:
    python display_centralized.py C:/CaptureData Gen3_BP_C RightEyeCamera 0
"""

import sys
import cv2
from data_loader import DataLoader

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    # Load data
    loader = DataLoader(data_path)
    
    if loader.format_type != 'centralized':
        print(f"Error: Data path contains {loader.format_type} format, not centralized format")
        sys.exit(1)
    
    # Get actor and camera names
    if len(sys.argv) >= 4:
        actor_name = sys.argv[2]
        camera_name = sys.argv[3]
        
        if actor_name not in loader.actors:
            print(f"Error: Actor '{actor_name}' not found")
            print(f"Available actors: {', '.join(loader.actors.keys())}")
            sys.exit(1)
        
        if camera_name not in loader.actors[actor_name]:
            print(f"Error: Camera '{camera_name}' not found on actor '{actor_name}'")
            print(f"Available cameras: {', '.join(loader.actors[actor_name])}")
            sys.exit(1)
    else:
        # Use first actor and first camera
        actor_name = list(loader.actors.keys())[0]
        camera_name = loader.actors[actor_name][0]
        print(f"Using first actor/camera: {actor_name}/{camera_name}")
    
    # Get starting frame
    start_frame = int(sys.argv[4]) if len(sys.argv) >= 5 else 0
    
    # Get frame count
    frame_count = loader.get_frame_count(actor_name, camera_name)
    print(f"Found {frame_count} frames for {actor_name}/{camera_name}")
    
    if frame_count == 0:
        print("No frames found!")
        sys.exit(1)
    
    # Display frames
    current_frame = start_frame
    
    print("\nControls:")
    print("  Space: Next frame")
    print("  Backspace: Previous frame")
    print("  Q or ESC: Quit")
    print()
    
    while True:
        try:
            # Load and display frame
            loader.display_frame(actor_name, camera_name, current_frame)
            
            # Wait for key
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # Space
                current_frame = min(current_frame + 1, frame_count - 1)
            elif key == 8:  # Backspace
                current_frame = max(current_frame - 1, 0)
            elif key == ord('s'):  # S - save current view
                output_name = f"{actor_name}_{camera_name}_frame_{current_frame:07d}.png"
                # Get current displayed image
                frame_data = loader.load_frame_exr(actor_name, camera_name, current_frame)
                rgb_display = loader.convert_rgb_for_display(frame_data['rgb'])
                cv2.imwrite(output_name, rgb_display)
                print(f"Saved: {output_name}")
            
            print(f"Frame: {current_frame}/{frame_count - 1}", end='\r')
            
        except FileNotFoundError as e:
            print(f"\nError loading frame {current_frame}: {e}")
            break
        except KeyboardInterrupt:
            break
    
    cv2.destroyAllWindows()
    print("\nDone!")

if __name__ == '__main__':
    main()
