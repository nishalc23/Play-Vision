"""
Video utility functions for sports highlight generator
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Optional, Generator
from pathlib import Path
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

def get_video_info(video_path: str) -> Dict:
    """
    Get comprehensive video information
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video metadata
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
        
        info = {
            'path': video_path,
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
            'bitrate': cap.get(cv2.CAP_PROP_BITRATE) if hasattr(cv2, 'CAP_PROP_BITRATE') else None,
        }
        
        # Calculate additional properties
        if info['fps'] > 0:
            info['duration'] = info['total_frames'] / info['fps']
        else:
            info['duration'] = 0
        
        info['aspect_ratio'] = info['width'] / info['height'] if info['height'] > 0 else 0
        info['resolution'] = f"{info['width']}x{info['height']}"
        info['file_size'] = os.path.getsize(video_path) if os.path.exists(video_path) else 0
        
        # Convert fourcc to readable format
        fourcc_bytes = info['fourcc'].to_bytes(4, byteorder='little')
        try:
            info['codec'] = fourcc_bytes.decode('ascii').rstrip('\x00')
        except:
            info['codec'] = 'unknown'
        
        cap.release()
        return info
        
    except Exception as e:
        logger.error(f"Failed to get video info for {video_path}: {e}")
        return {}

def validate_video_file(video_path: str) -> bool:
    """
    Validate if video file can be opened and read
    
    Args:
        video_path: Path to video file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        if not os.path.exists(video_path):
            return False
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # Try to read first frame
        ret, frame = cap.read()
        cap.release()
        
        return ret and frame is not None
        
    except Exception as e:
        logger.error(f"Video validation failed for {video_path}: {e}")
        return False

def extract_frames(
    video_path: str, 
    start_frame: int = 0, 
    end_frame: Optional[int] = None,
    step: int = 1
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Extract frames from video with generator for memory efficiency
    
    Args:
        video_path: Path to video file
        start_frame: Starting frame number
        end_frame: Ending frame number (None for all frames)
        step: Frame step size
    
    Yields:
        Tuple of (frame_number, frame_array)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None:
            end_frame = total_frames
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_number = start_frame
        while frame_number < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            if (frame_number - start_frame) % step == 0:
                yield frame_number, frame
            
            frame_number += 1
            
            # Skip frames if step > 1
            if step > 1:
                for _ in range(step - 1):
                    ret, _ = cap.read()
                    if not ret:
                        break
                    frame_number += 1
                if not ret:
                    break
    
    finally:
        cap.release()

def extract_frame_at_time(video_path: str, timestamp: float) -> Optional[np.ndarray]:
    """
    Extract single frame at specific timestamp
    
    Args:
        video_path: Path to video file
        timestamp: Time in seconds
    
    Returns:
        Frame array or None if failed
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        return frame if ret else None
        
    except Exception as e:
        logger.error(f"Failed to extract frame at {timestamp}s: {e}")
        return None

def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculate difference between two frames
    
    Args:
        frame1: First frame
        frame2: Second frame
    
    Returns:
        Difference score (higher = more different)
    """
    try:
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1
            
        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Return mean difference
        return np.mean(diff)
        
    except Exception as e:
        logger.error(f"Frame difference calculation failed: {e}")
        return 0.0

def detect_scene_changes(
    video_path: str, 
    threshold: float = 30.0,
    sample_rate: int = 1
) -> List[Tuple[int, float]]:
    """
    Detect scene changes in video based on frame differences
    
    Args:
        video_path: Path to video file
        threshold: Difference threshold for scene change
        sample_rate: Process every Nth frame
    
    Returns:
        List of (frame_number, difference_score) for scene changes
    """
    scene_changes = []
    prev_frame = None
    
    try:
        for frame_num, frame in extract_frames(video_path, step=sample_rate):
            if prev_frame is not None:
                diff_score = calculate_frame_difference(prev_frame, frame)
                
                if diff_score > threshold:
                    scene_changes.append((frame_num, diff_score))
            
            prev_frame = frame
        
        logger.info(f"Detected {len(scene_changes)} scene changes")
        return scene_changes
        
    except Exception as e:
        logger.error(f"Scene change detection failed: {e}")
        return []

def crop_frame(
    frame: np.ndarray, 
    region: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Crop frame to specified region
    
    Args:
        frame: Input frame
        region: (x, y, width, height)
    
    Returns:
        Cropped frame
    """
    x, y, width, height = region
    return frame[y:y+height, x:x+width]

def resize_frame(
    frame: np.ndarray, 
    size: Tuple[int, int], 
    maintain_aspect: bool = True
) -> np.ndarray:
    """
    Resize frame to specified size
    
    Args:
        frame: Input frame
        size: Target (width, height)
        maintain_aspect: Whether to maintain aspect ratio
    
    Returns:
        Resized frame
    """
    if maintain_aspect:
        h, w = frame.shape[:2]
        target_w, target_h = size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad to target size if needed
        if new_w < target_w or new_h < target_h:
            # Create black canvas
            canvas = np.zeros((target_h, target_w, frame.shape[2]), dtype=frame.dtype)
            
            # Center the resized image
            start_y = (target_h - new_h) // 2
            start_x = (target_w - new_w) // 2
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
            
            return canvas
        else:
            return resized
    else:
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

def enhance_frame(
    frame: np.ndarray,
    brightness: float = 0,
    contrast: float = 1.0,
    saturation: float = 1.0
) -> np.ndarray:
    """
    Enhance frame with brightness, contrast, and saturation adjustments
    
    Args:
        frame: Input frame
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast multiplier (0.5 to 3.0)
        saturation: Saturation multiplier (0.0 to 2.0)
    
    Returns:
        Enhanced frame
    """
    enhanced = frame.copy().astype(np.float32)
    
    # Brightness adjustment
    enhanced = enhanced + brightness
    
    # Contrast adjustment
    enhanced = enhanced * contrast
    
    # Saturation adjustment (if color image)
    if len(frame.shape) == 3 and saturation != 1.0:
        # Convert to HSV for saturation adjustment
        hsv = cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    
    # Clip values to valid range
    enhanced = np.clip(enhanced, 0, 255)
    
    return enhanced.astype(np.uint8)
