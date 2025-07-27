import cv2

def stream_camera(camera_index=1):
    """
    Stream video from OpenCV camera with specified index.
    
    Args:
        camera_index (int): Index of the camera to use (default: 1)
    """
    # Initialize the camera
    cap = cv2.VideoCapture(camera_index)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}")
        return
    
    print(f"Streaming from camera {camera_index}. Press 'q' to quit.")
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # If frame is read correctly, ret is True
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break
            
            # Display the resulting frame
            cv2.imshow(f'Camera {camera_index} Stream', frame)
            
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStream interrupted by user")
    
    finally:
        # Release the camera and close windows
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")

if __name__ == "__main__":
    stream_camera(camera_index=1)
