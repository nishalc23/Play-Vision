"""
Object detection module using YOLO for sports video analysis
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Please install ultralytics: pip install ultralytics")

from config.settings import (
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD, SPORTS_RELEVANT_CLASSES,
    YOLO_CLASSES, DEFAULT_MODEL_PATH
)

logger = logging.getLogger(__name__)

class SportsDetector:
    """Object detector specialized for sports videos"""
    
    def __init__(
        self, 
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        device: Optional[str] = None
    ):
        """
        Initialize the sports detector
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for non-maximum suppression
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self._load_model()
        self._setup_device()
        
        logger.info(f"SportsDetector initialized with {model_path} on {self.device}")
    
    def _load_model(self) -> YOLO:
        """Load YOLO model with error handling"""
        try:
            if not Path(self.model_path).exists() and not self.model_path.startswith('yolo'):
                logger.warning(f"Model file not found: {self.model_path}")
                logger.info("Downloading default YOLOv8 model...")
                self.model_path = "yolov8n.pt"
            
            model = YOLO(self.model_path)
            logger.info(f"Loaded model: {self.model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_path}: {e}")
            raise
    
    def _setup_device(self):
        """Setup device for inference"""
        try:
            self.model.to(self.device)
            logger.info(f"Model moved to device: {self.device}")
        except Exception as e:
            logger.warning(f"Could not move model to {self.device}, using CPU: {e}")
            self.device = 'cpu'
    
    def detect_frame(self, frame: np.ndarray, filter_sports: bool = True) -> Dict:
        """
        Detect objects in a single frame
        
        Args:
            frame: Input frame as numpy array
            filter_sports: Whether to filter for sports-relevant classes only
        
        Returns:
            Dictionary containing detection results
        """
        try:
            # Run inference
            results = self.model(
                frame, 
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Parse results
            detections = self._parse_results(results[0], filter_sports)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return self._empty_detection_result()
    
    def _parse_results(self, result, filter_sports: bool) -> Dict:
        """Parse YOLO results into structured format"""
        detections = {
            'players': [],
            'balls': [],
            'equipment': [],
            'all_objects': [],
            'frame_info': {
                'total_detections': 0,
                'player_count': 0,
                'ball_count': 0,
                'equipment_count': 0
            }
        }
        
        if result.boxes is None:
            return detections
        
        boxes = result.boxes.cpu()
        
        for i, box in enumerate(boxes):
            # Extract detection information
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # Get class name
            class_name = self.model.names.get(class_id, f"class_{class_id}")
            
            # Skip if filtering for sports and class not relevant
            if filter_sports and class_name not in SPORTS_RELEVANT_CLASSES:
                continue
            
            # Create detection object
            detection = {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence,
                'class': class_name,
                'class_id': class_id,
                'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                'area': int((x2 - x1) * (y2 - y1)),
                'width': int(x2 - x1),
                'height': int(y2 - y1)
            }
            
            # Categorize detection
            if class_name == 'person':
                detections['players'].append(detection)
                detections['frame_info']['player_count'] += 1
                
            elif class_name == 'sports ball':
                detections['balls'].append(detection)
                detections['frame_info']['ball_count'] += 1
                
            elif class_name in ['baseball bat', 'baseball glove', 'tennis racket']:
                detections['equipment'].append(detection)
                detections['frame_info']['equipment_count'] += 1
            
            detections['all_objects'].append(detection)
            detections['frame_info']['total_detections'] += 1
        
        return detections
    
    def _empty_detection_result(self) -> Dict:
        """Return empty detection result structure"""
        return {
            'players': [],
            'balls': [],
            'equipment': [],
            'all_objects': [],
            'frame_info': {
                'total_detections': 0,
                'player_count': 0,
                'ball_count': 0,
                'equipment_count': 0
            }
        }
    
    def detect_batch(
        self, 
        frames: List[np.ndarray], 
        filter_sports: bool = True
    ) -> List[Dict]:
        """
        Detect objects in batch of frames
        
        Args:
            frames: List of frames to process
            filter_sports: Whether to filter for sports-relevant classes
        
        Returns:
            List of detection results for each frame
        """
        try:
            # Run batch inference
            results = self.model(
                frames,
                conf=self
