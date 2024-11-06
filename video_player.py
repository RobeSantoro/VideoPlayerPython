import cv2
import numpy as np
import random
import os
import time
from typing import Dict, List, NamedTuple, Tuple
from queue import Queue
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor
import numba

# Constants
VIDEOS_PER_POSITION = 2
PATH_TO_VIDEO = "videos"
TARGET_WIDTH = 720
TARGET_HEIGHT = 1280
BUFFER_SIZE = 5
MAX_WORKERS = 4

class VideoPosition(NamedTuple):
    x: int
    y: int
    scale: float
    rotation: float  # in degrees

# Updated POSITION_MAP with rotation and scale
POSITION_MAP = {
    ord('\\'): VideoPosition(-360, -640, 1.5, 45),    # 50% larger, rotated 45°
    ord('1'): VideoPosition(300, 0, 1.0, 0),          # Normal size, no rotation
    ord('2'): VideoPosition(600, 0, 1.2, 30),         # 20% larger, rotated 30°
    ord('3'): VideoPosition(800, 0, 0.8, -15),        # 20% smaller, rotated -15°
    ord('4'): VideoPosition(1024, 0, 1.5, 90),        # 50% larger, rotated 90°
    ord('5'): VideoPosition(1280, 0, 0.7, 60),        # 30% smaller, rotated 60°
    ord('6'): VideoPosition(1600, 0, 1.3, -45),       # 30% larger, rotated -45°
}

@numba.jit(nopython=True, parallel=True)
def fast_blend(background: np.ndarray, foreground: np.ndarray, x: int, y: int) -> np.ndarray:
    """Optimized blending using Numba"""
    fg_h, fg_w = foreground.shape[:2]
    bg_h, bg_w = background.shape[:2]
    
    src_x = max(0, -x)
    src_y = max(0, -y)
    dst_x = max(0, x)
    dst_y = max(0, y)
    
    visible_w = min(fg_w - src_x, bg_w - dst_x)
    visible_h = min(fg_h - src_y, bg_h - dst_y)
    
    if visible_w <= 0 or visible_h <= 0:
        return background
        
    for i in numba.prange(visible_h):
        for j in range(visible_w):
            for c in range(3):
                background[dst_y + i, dst_x + j, c] = foreground[src_y + i, src_x + j, c]
    
    return background

class VideoBuffer:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        # Enable hardware acceleration if available
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)
        
        self.frames = Queue(maxsize=BUFFER_SIZE)
        self.is_active = True
        self.lock = Lock()
        self.thread = Thread(target=self._buffer_frames, daemon=True)
        self.thread.start()

    def _buffer_frames(self):
        while self.is_active:
            if self.frames.qsize() < BUFFER_SIZE:
                ret, frame = self.cap.read()
                if ret:
                    # Use INTER_LINEAR for better performance
                    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), 
                                    interpolation=cv2.INTER_LINEAR)
                    self.frames.put(frame)
                else:
                    self.is_active = False
                    self.cap.release()
            else:
                time.sleep(0.001)

    def get_frame(self):
        try:
            return self.frames.get_nowait()
        except:
            return None

    def release(self):
        self.is_active = False
        self.thread.join(timeout=1.0)
        with self.lock:
            if self.cap is not None:
                self.cap.release()

class VideoPlayer:
    def __init__(self):
        self.active_videos: Dict[int, List[Dict]] = {}
        self.available_files: List[str] = []
        self.used_files: List[str] = []
        self.current_layer = 0
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        self.fps = 0
        self.fps_smoothing = 0.95
        
        # Increase thread pool size based on CPU cores
        self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())
        
        # Pre-allocate buffers
        self.background_buffer = np.zeros((1440, 2560, 3), dtype=np.uint8)
        self.rotation_matrix = None
        self.frame_lock = Lock()
        
        self.load_video_files()

    @staticmethod
    def rotate_and_scale(frame: np.ndarray, angle: float, scale: float) -> np.ndarray:
        """Rotate and scale the frame efficiently"""
        if angle == 0 and scale == 1.0:
            return frame
            
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        
        # Combine rotation and scaling in a single matrix
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Calculate new dimensions
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust translation
        matrix[0, 2] += (new_width / 2) - center[0]
        matrix[1, 2] += (new_height / 2) - center[1]
        
        # Use INTER_LINEAR for better performance
        return cv2.warpAffine(frame, matrix, (new_width, new_height),
                            flags=cv2.INTER_LINEAR)

    def process_video_frame(self, video_info):
        """Process a single video frame with rotation and scaling"""
        frame = video_info['buffer'].get_frame()
        if frame is not None:
            position = video_info['position']
            # Apply rotation and scaling
            frame = self.rotate_and_scale(frame, position.rotation, position.scale)
            video_info['last_frame'] = frame
            return frame, (position.x, position.y), video_info['layer']
        return video_info['last_frame'], (video_info['position'].x, video_info['position'].y), video_info['layer']

    def start_video(self, key: int, position: VideoPosition):
        if key not in self.active_videos:
            self.active_videos[key] = []
        
        if len(self.active_videos[key]) >= VIDEOS_PER_POSITION:
            oldest_video = self.active_videos[key].pop(0)
            if 'buffer' in oldest_video:
                oldest_video['buffer'].release()

        video_path = self.get_random_video()
        video_buffer = VideoBuffer(video_path)
        
        self.current_layer += 1
        new_video = {
            'buffer': video_buffer,
            'position': position,
            'layer': self.current_layer,
            'last_frame': None
        }
        self.active_videos[key].append(new_video)

    def update(self) -> np.ndarray:
        # Use numpy's optimized fill
        np.copyto(self.background_buffer, 0)
        
        # Collect all active videos
        all_videos = []
        for videos in self.active_videos.values():
            all_videos.extend((video, idx) for idx, video in enumerate(videos))
        
        # Process frames in parallel
        frame_futures = []
        for video_info, _ in sorted(all_videos, key=lambda x: x[0]['layer']):
            future = self.thread_pool.submit(self.process_video_frame, video_info)
            frame_futures.append(future)
        
        # Collect and blend frames in order
        for future in frame_futures:
            frame, position, _ = future.result()
            if frame is not None:
                self.background_buffer = fast_blend(
                    self.background_buffer, frame, position[0], position[1])

        self.calculate_fps()
        return self.draw_fps(self.background_buffer)

    def load_video_files(self):
        if os.path.exists(PATH_TO_VIDEO):
            self.available_files = [f for f in os.listdir(PATH_TO_VIDEO) 
                                  if f.endswith(('.mp4', '.avi', '.mov'))]
        if not self.available_files:
            raise Exception(f"No video files found in {PATH_TO_VIDEO}")

    def get_random_video(self) -> str:
        if not self.available_files:
            self.available_files = self.used_files
            self.used_files = []
        
        video_file = random.choice(self.available_files)
        self.available_files.remove(video_file)
        self.used_files.append(video_file)
        return os.path.join(PATH_TO_VIDEO, video_file)

    def blend_frames(self, background: np.ndarray, foreground: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """Modified blend_frames to handle partial frame rendering"""
        x, y = position
        fg_h, fg_w = foreground.shape[:2]
        bg_h, bg_w = background.shape[:2]

        # Calculate visible region
        src_x = max(0, -x)
        src_y = max(0, -y)
        dst_x = max(0, x)
        dst_y = max(0, y)
        
        # Calculate width and height of visible region
        visible_w = min(fg_w - src_x, bg_w - dst_x)
        visible_h = min(fg_h - src_y, bg_h - dst_y)

        # Skip if no visible region
        if visible_w <= 0 or visible_h <= 0:
            return background

        # Extract visible region of foreground
        fg_visible = foreground[src_y:src_y + visible_h, src_x:src_x + visible_w]

        # Blend visible region
        if foreground.shape[2] == 4:
            alpha = fg_visible[:, :, 3:] / 255.0
            rgb = fg_visible[:, :, :3]
            roi = background[dst_y:dst_y + visible_h, dst_x:dst_x + visible_w]
            np.multiply(1 - alpha, roi, out=roi)
            np.add(roi, alpha * rgb, out=roi)
        else:
            background[dst_y:dst_y + visible_h, dst_x:dst_x + visible_w] = fg_visible

        return background

    def calculate_fps(self):
        self.curr_frame_time = time.time()
        if self.prev_frame_time == 0:
            self.prev_frame_time = self.curr_frame_time
            return
            
        inst_fps = 1 / (self.curr_frame_time - self.prev_frame_time)
        if self.fps == 0:
            self.fps = inst_fps
        else:
            self.fps = (self.fps_smoothing * self.fps + 
                       (1 - self.fps_smoothing) * inst_fps)
        self.prev_frame_time = self.curr_frame_time

    def draw_fps(self, frame: np.ndarray) -> np.ndarray:
        fps_text = f"FPS: {self.fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        color = (255, 255, 255)
        
        (text_width, text_height), _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
        cv2.rectangle(frame, (10, 10), (20 + text_width, 20 + text_height), 
                     (0, 0, 0), -1)
        cv2.putText(frame, fps_text, (15, 15 + text_height), font, font_scale, 
                   color, thickness, cv2.LINE_AA)
        return frame

    def run(self):
        cv2.namedWindow('Video Player', cv2.WINDOW_NORMAL)
        
        try:
            while True:
                frame = self.update()
                cv2.imshow('Video Player', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key in POSITION_MAP:
                    self.start_video(key, POSITION_MAP[key])
        finally:
            # Cleanup
            for video_list in self.active_videos.values():
                for video in video_list:
                    if 'buffer' in video:
                        video['buffer'].release()
            
            self.thread_pool.shutdown(wait=False)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    player = VideoPlayer()
    player.run()