import cv2
from data_loader import DataLoader
import time

def main(argv):
    if len(argv) >= 3:
        folder_name = argv[1]
        camera_name = argv[2]
    else:
        print("You must provide <folder_path> <camera name>")
        print("You provided:\n", argv)
        exit()

    fps = 30
    index = 0
    playing = False

    dataLoader = DataLoader(folder_name)

    while True:
        try:
            dataLoader.display_raw_stack(camera_name, index)
        except Exception as e:
            print("Exception at index " + str(index) + " - " + str(e))
            exit()

        while True:
            key = cv2.waitKey(int(1000.0/fps)) # pauses for at most (1/fps) seconds before fetching next image
            if key == 27 or key == ord('q'): # if ESC is pressed, exit loop
                cv2.destroyAllWindows()
                exit()
            elif key == 32: # if space is pressed, toggle playing
                playing = not playing
                break
            elif key == ord('p'): # if p is next go to previous
                index -= 1
                if index < 0: index = 0
                break
            elif key == ord('n'): # if n is pressed go to next
                index += 1
                break
            elif playing:
                index += 1
                break

if __name__ == "__main__":
    import sys
    main(sys.argv)
