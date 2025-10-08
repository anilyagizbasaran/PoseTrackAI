"""
YOLO Pose Detection - RTSP Camera Module
Specialized configuration and management for RTSP camera streams
"""

import cv2
from log import log_with_timestamp


class RTSPCamera:
    """Manage RTSP camera stream"""
    
    def __init__(self, rtsp_url, width=640, height=480):
        """
        Initialize RTSP camera
        
        Args:
            rtsp_url: RTSP stream URL
            width: Desired frame width
            height: Desired frame height
        """
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.cap = None
        
    def connect(self):
        """Connect to camera"""
        log_with_timestamp(f"Connecting to RTSP camera: {self.rtsp_url}", "CAMERA")
        
        try:
            # OpenCV-FFMPEG options (must be set before VideoCapture)
            # Timeout settings (in microseconds)
            import os
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|buffer_size;1024000|max_delay;500000'
            
            # Special settings for RTSP - FFMPEG backend required
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            # AKICILIK için ayarlar
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimum buffer - eski frame'leri atla
            # FPS limitini KALDIRDIK - kameranın native FPS'ini kullansın
            
            if not self.cap.isOpened():
                log_with_timestamp(f"Could not connect to RTSP camera!", "ERROR")
                log_with_timestamp(f"URL: {self.rtsp_url}", "ERROR")
                return False
        except Exception as e:
            log_with_timestamp(f"RTSP connection error: {e}", "ERROR")
            return False
        
        # Set resolution (some cameras may not support this)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Read test frame to get actual resolution
        log_with_timestamp("Reading test frame...", "CAMERA")
        success, test_frame = self.cap.read()
        
        if not success or test_frame is None:
            log_with_timestamp(f"Could not read frame from RTSP stream!", "ERROR")
            self.cap.release()
            return False
        
        # Get actual resolution from frame
        actual_height, actual_width = test_frame.shape[:2]
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Use default value if FPS is abnormal
        if actual_fps <= 0 or actual_fps > 120:
            actual_fps = 25
        
        log_with_timestamp(f"RTSP camera connected!", "CAMERA")
        log_with_timestamp(f"Resolution: {actual_width}x{actual_height}", "CAMERA")
        log_with_timestamp(f"FPS: {actual_fps}", "CAMERA")
        
        return True
    
    def read_frame(self, skip_buffered=True):
        """
        Read frame from camera
        
        Args:
            skip_buffered: Eski buffer'daki frame'leri atla, en güncel frame'i al (akıcılık için)
        
        Returns:
            success: Was frame read successfully?
            frame: Read frame (BGR)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        try:
            # AKICILIK İÇİN: Buffer'daki eski frame'leri atla, en güncel frame'i al
            if skip_buffered:
                # Eski frame'leri temizle - sadece en son frame'i oku
                for _ in range(3):  # Buffer'daki eski frame'leri atla
                    self.cap.grab()
            
            # En güncel frame'i oku
            success, frame = self.cap.read()
            
            # Frame validation
            if success and frame is not None and frame.size > 0:
                return True, frame
            else:
                return False, None
                
        except Exception as e:
            log_with_timestamp(f"Frame read error: {e}", "WARNING")
            return False, None
    
    def reconnect(self, max_attempts=3, wait_time=5):
        """
        Reconnect if connection is lost
        
        Args:
            max_attempts: Maximum number of attempts
            wait_time: Wait time between attempts (seconds)
        
        Returns:
            bool: Was successful?
        """
        log_with_timestamp("Camera connection lost, reconnecting...", "CAMERA")
        
        # Safely close old connection
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        except Exception as e:
            log_with_timestamp(f"Connection close error: {e}", "WARNING")
        
        # Try to reconnect
        import time
        for attempt in range(1, max_attempts + 1):
            log_with_timestamp(f"Attempt {attempt}/{max_attempts}...", "CAMERA")
            
            # Short wait (except for first attempt)
            if attempt > 1:
                time.sleep(wait_time)
            
            if self.connect():
                log_with_timestamp("Reconnection successful!", "CAMERA")
                return True
        
        log_with_timestamp("Reconnection failed!", "ERROR")
        return False
    
    def release(self):
        """Close camera connection"""
        if self.cap is not None:
            self.cap.release()
            log_with_timestamp("Camera connection closed", "CAMERA")
    
    def get_properties(self):
        """
        Get camera properties
        
        Returns:
            dict: Camera properties
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # If resolution is 0, get real values by reading a frame
        if width == 0 or height == 0:
            success, frame = self.cap.read()
            if success and frame is not None:
                height, width = frame.shape[:2]
        
        # Use default value if FPS is abnormal
        if fps <= 0 or fps > 120:
            fps = 25
        
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'backend': self.cap.getBackendName()
        }
