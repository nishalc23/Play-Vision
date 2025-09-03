"""
Video analysis module for detecting sports highlights and events
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
from datetime import datetime

from .detector import SportsDetector
from config.settings import (
    BALL_VELOCITY_THRESHOLD, GOAL_AREA_THRESHOLD, MERGE_WINDOW,
    MIN_EVENT_SEPARATION, DEFAULT_CLIP_DURATION, get_sport_config,
    EVENT_CONFIDENCE_WEIGHTS
)

logger = logging.getLogger(__name__)

class SportsAnalyzer:
    """Main analyzer for detecting sports highlights in videos"""
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        sport_type: str = "basketball"
    ):
        """
        Initialize the sports analyzer
        
        Args:
            model_path: Path to YOLO model
            confidence_threshold: Detection confidence threshold
            sport_type: Type of sport to analyze
        """
        self.detector = SportsDetector(model_path, confidence_threshold)
        self.sport_type = sport_type.lower()
        self.sport_config = get_sport_config(self.sport_type)
        
        self.video_path = None
        self.video_info = {}
        self.analysis_history = []
        
        logger.info(f"SportsAnalyzer initialized for {sport_type}")
    
    def load_video(self, video_path: str) -> bool:
        """
        Load video file and extract metadata
        
        Args:
            video_path: Path to video file
        
        Returns:
            Success status
        """
        try:
            self.video_path = Path(video_path)
            if not self.video_path.exists():
                logger.error(f"Video file not found: {video_path}")
                return False
            
            # Extract video information
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return False
            
            self.video_info = {
                'path': str(video_path),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                'loaded_at': datetime.now().isoformat()
            }
            
            cap.release()
            
            logger.info(f"Video loaded: {self.video_info['width']}x{self.video_info['height']}, "
                       f"{self.video_info['fps']} FPS, {self.video_info['duration']:.1f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load video: {e}")
            return False
    
    def analyze_video(
        self,
        sample_rate: int = 2,
        progress_callback: Optional[callable] = None
    ) -> List[Dict]:
        """
        Analyze entire video for highlights
        
        Args:
            sample_rate: Process every Nth frame
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of detected highlight events
        """
        if not self.video_path:
            raise ValueError("No video loaded. Call load_video() first.")
        
        logger.info(f"Starting analysis of {self.video_path} for {self.sport_type}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        highlights = []
        frame_number = 0
        prev_detections = None
        prev_ball_position = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every sample_rate frames
                if frame_number % sample_rate == 0:
                    # Detect objects
                    detections = self.detector.detect_frame(frame)
                    
                    # Analyze for events
                    events = self.analyze_frame(
                        detections, 
                        prev_detections,
                        frame_number,
                        prev_ball_position
                    )
                    
                    # Process any detected events
                    for event in events:
                        if event['is_highlight']:
                            highlight = self.create_highlight_event(
                                event, frame_number, detections
                            )
                            highlights.append(highlight)
                            logger.debug(f"Highlight at {highlight['timestamp']:.1f}s: {event['reason']}")
                    
                    # Update previous state
                    prev_detections = detections
                    ball = self.detector.get_primary_ball(detections)
                    prev_ball_position = ball['center'] if ball else None
                
                frame_number += 1
                
                # Progress callback
                if progress_callback and frame_number % 300 == 0:
                    progress = frame_number / self.video_info['total_frames']
                    progress_callback(progress)
        
        finally:
            cap.release()
        
        # Post-process highlights
        highlights = self.merge_nearby_highlights(highlights)
        highlights = self.filter_highlights(highlights)
        
        logger.info(f"Analysis complete. Found {len(highlights)} highlights")
        self.analysis_history.append({
            'timestamp': datetime.now().isoformat(),
            'video_path': str(self.video_path),
            'sport_type': self.sport_type,
            'highlights_found': len(highlights),
            'sample_rate': sample_rate
        })
        
        return highlights
    
    def analyze_frame(
        self,
        detections: Dict,
        prev_detections: Optional[Dict],
        frame_number: int,
        prev_ball_position: Optional[Tuple[int, int]]
    ) -> List[Dict]:
        """
        Analyze a single frame for highlight events
        
        Args:
            detections: Current frame detections
            prev_detections: Previous frame detections
            frame_number: Current frame number
            prev_ball_position: Previous ball position
        
        Returns:
            List of detected events
        """
        events = []
        
        # Get current ball
        current_ball = self.detector.get_primary_ball(detections)
        current_ball_position = current_ball['center'] if current_ball else None
        
        # Event detection heuristics
        events.extend(self._detect_ball_events(
            current_ball, prev_ball_position, current_ball_position, frame_number
        ))
        
        events.extend(self._detect_player_events(
            detections, prev_detections, frame_number
        ))
        
        events.extend(self._detect_scene_events(
            detections, prev_detections, frame_number
        ))
        
        return events
    
    def _detect_ball_events(
        self,
        current_ball: Optional[Dict],
        prev_ball_position: Optional[Tuple[int, int]],
        current_ball_position: Optional[Tuple[int, int]],
        frame_number: int
    ) -> List[Dict]:
        """Detect ball-related highlight events"""
        events = []
        
        if not current_ball:
            # Ball disappeared - check if it was in goal area
            if prev_ball_position and self._is_in_goal_area(prev_ball_position):
                events.append({
                    'is_highlight': True,
                    'event_type': 'ball_disappeared_goal',
                    'confidence': EVENT_CONFIDENCE_WEIGHTS['ball_disappeared'],
                    'reason': 'Ball disappeared near goal area',
                    'frame_number': frame_number,
                    'ball_position': prev_ball_position
                })
            return events
        
        # Fast ball movement
        if prev_ball_position and current_ball_position:
            velocity = self._calculate_ball_velocity(prev_ball_position, current_ball_position)
            
            # High velocity event
            if velocity > self.sport_config['ball_velocity_threshold']:
                confidence = min(0.9, velocity / 100) * EVENT_CONFIDENCE_WEIGHTS['fast_ball_movement']
                
                events.append({
                    'is_highlight': True,
                    'event_type': 'fast_ball_movement',
                    'confidence': confidence,
                    'reason': f'Fast ball movement: {velocity:.1f} pixels/frame',
                    'frame_number': frame_number,
                    'ball_velocity': velocity,
                    'ball_position': current_ball_position
                })
            
            # Ball in goal area with movement
            if self._is_in_goal_area(current_ball_position):
                confidence = EVENT_CONFIDENCE_WEIGHTS['ball_in_goal_area']
                if velocity > self.sport_config['ball_velocity_threshold'] * 0.5:
                    confidence *= 1.2  # Boost confidence for moving ball in goal
                
                events.append({
                    'is_highlight': True,
                    'event_type': 'ball_in_goal_area',
                    'confidence': min(1.0, confidence),
                    'reason': f'Ball in goal area (velocity: {velocity:.1f})',
                    'frame_number': frame_number,
                    'ball_velocity': velocity,
                    'ball_position': current_ball_position
                })
        
        return events
    
    def _detect_player_events(
        self,
        detections: Dict,
        prev_detections: Optional[Dict],
        frame_number: int
    ) -> List[Dict]:
        """Detect player-related highlight events"""
        events = []
        
        player_count = len(detections.get('players', []))
        expected_range = self.sport_config['player_count_range']
        
        # High player density (more players than usual in frame)
        if player_count > expected_range[1] * 0.8:  # 80% of max expected
            confidence = min(1.0, player_count / expected_range[1]) * EVENT_CONFIDENCE_WEIGHTS['high_player_density']
            
            events.append({
                'is_highlight': True,
                'event_type': 'high_player_density',
                'confidence': confidence,
                'reason': f'High player activity: {player_count} players detected',
                'frame_number': frame_number,
                'player_count': player_count
            })
        
        # Sudden player movement (if we have previous frame)
        if prev_detections:
            prev_players = prev_detections.get('players', [])
            if len(prev_players) > 0 and player_count > 0:
                movement_score = self._calculate_player_movement(
                    prev_players, detections.get('players', [])
                )
                
                if movement_score > 100:  # Threshold for significant movement
                    confidence = min(1.0, movement_score / 200) * 0.4
                    
                    events.append({
                        'is_highlight': True,
                        'event_type': 'rapid_player_movement',
                        'confidence': confidence,
                        'reason': f'Rapid player movement detected: {movement_score:.1f}',
                        'frame_number': frame_number,
                        'movement_score': movement_score
                    })
        
        return events
    
    def _detect_scene_events(
        self,
        detections: Dict,
        prev_detections: Optional[Dict],
        frame_number: int
    ) -> List[Dict]:
        """Detect scene-level highlight events"""
        events = []
        
        if not prev_detections:
            return events
        
        # Sudden change in detection count (scene change, crowd reaction)
        current_total = len(detections.get('all_objects', []))
        prev_total = len(prev_detections.get('all_objects', []))
        
        if prev_total > 0:
            change_ratio = abs(current_total - prev_total) / prev_total
            
            if change_ratio > 0.5:  # 50% change threshold
                confidence = min(1.0, change_ratio) * EVENT_CONFIDENCE_WEIGHTS['rapid_scene_change']
                
                events.append({
                    'is_highlight': True,
                    'event_type': 'scene_change',
                    'confidence': confidence,
                    'reason': f'Rapid scene change: {change_ratio:.2f} detection ratio change',
                    'frame_number': frame_number,
                    'detection_change': change_ratio
                })
        
        return events
    
    def _is_in_goal_area(self, position: Tuple[int, int]) -> bool:
        """Check if position is in goal/scoring area based on sport type"""
        x, y = position
        width = self.video_info['width']
        height = self.video_info['height']
        
        threshold = self.sport_config['goal_area_threshold']
        
        if self.sport_type == 'basketball':
            # Goals at top and bottom
            goal_height = int(height * threshold)
            return y < goal_height or y > (height - goal_height)
        
        elif self.sport_type in ['soccer', 'football']:
            # Goals at left and right
            goal_width = int(width * threshold)
            return x < goal_width or x > (width - goal_width)
        
        # Default: assume basketball-style
        goal_height = int(height * threshold)
        return y < goal_height or y > (height - goal_height)
    
    def _calculate_ball_velocity(
        self, 
        prev_pos: Tuple[int, int], 
        curr_pos: Tuple[int, int]
    ) -> float:
        """Calculate ball velocity between frames"""
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        return np.sqrt(dx**2 + dy**2)
    
    def _calculate_player_movement(
        self, 
        prev_players: List[Dict], 
        curr_players: List[Dict]
    ) -> float:
        """Calculate overall player movement between frames"""
        if not prev_players or not curr_players:
            return 0.0
        
        # Simple approach: compare average positions
        prev_centers = [p['center'] for p in prev_players]
        curr_centers = [p['center'] for p in curr_players]
        
        prev_avg_x = np.mean([c[0] for c in prev_centers])
        prev_avg_y = np.mean([c[1] for c in prev_centers])
        curr_avg_x = np.mean([c[0] for c in curr_centers])
        curr_avg_y = np.mean([c[1] for c in curr_centers])
        
        movement = np.sqrt((curr_avg_x - prev_avg_x)**2 + (curr_avg_y - prev_avg_y)**2)
        
        # Also consider spread change
        prev_spread = np.std([c[0] for c in prev_centers]) + np.std([c[1] for c in prev_centers])
        curr_spread = np.std([c[0] for c in curr_centers]) + np.std([c[1] for c in curr_centers])
        spread_change = abs(curr_spread - prev_spread)
        
        return movement + spread_change * 0.5
    
    def create_highlight_event(
        self, 
        event: Dict, 
        frame_number: int, 
        detections: Dict
    ) -> Dict:
        """Create a structured highlight event"""
        timestamp = frame_number / self.video_info['fps']
        
        highlight = {
            'frame_number': frame_number,
            'timestamp': timestamp,
            'duration': DEFAULT_CLIP_DURATION,
            'event_type': event['event_type'],
            'confidence': event['confidence'],
            'reason': event['reason'],
            'sport_type': self.sport_type,
            'detections_summary': {
                'players': len(detections.get('players', [])),
                'balls': len(detections.get('balls', [])),
                'total_objects': len(detections.get('all_objects', []))
            }
        }
        
        # Add event-specific data
        for key in ['ball_position', 'ball_velocity', 'player_count', 'movement_score', 'detection_change']:
            if key in event:
                highlight[key] = event[key]
        
        return highlight
    
    def merge_nearby_highlights(
        self, 
        highlights: List[Dict], 
        merge_window: float = MERGE_WINDOW
    ) -> List[Dict]:
        """Merge highlights that are close together in time"""
        if not highlights:
            return []
        
        # Sort by timestamp
        highlights.sort(key=lambda x: x['timestamp'])
        
        merged = []
        current_group = [highlights[0]]
        
        for highlight in highlights[1:]:
            time_diff = highlight['timestamp'] - current_group[-1]['timestamp']
            
            if time_diff <= merge_window:
                current_group.append(highlight)
            else:
                # Process current group
                merged_highlight = self._merge_highlight_group(current_group)
                merged.append(merged_highlight)
                current_group = [highlight]
        
        # Process final group
        merged_highlight = self._merge_highlight_group(current_group)
        merged.append(merged_highlight)
        
        logger.info(f"Merged {len(highlights)} highlights into {len(merged)} clips")
        return merged
    
    def _merge_highlight_group(self, highlights: List[Dict]) -> Dict:
        """Merge a group of highlights into a single event"""
        if len(highlights) == 1:
            return highlights[0]
        
        # Calculate merged properties
        start_time = highlights[0]['timestamp']
        end_time = max(h['timestamp'] + h['duration'] for h in highlights)
        max_confidence = max(h['confidence'] for h in highlights)
        
        # Determine primary event type
        event_types = [h['event_type'] for h in highlights]
        primary_type = max(set(event_types), key=event_types.count)
        
        merged = {
            'frame_number': highlights[0]['frame_number'],
            'timestamp': start_time,
            'duration': end_time - start_time,
            'event_type': f'merged_{primary_type}',
            'confidence': max_confidence,
            'reason': f'Merged sequence of {len(highlights)} events',
            'sport_type': self.sport_type,
            'sub_events': highlights,
            'merged_event_types': list(set(event_types))
        }
        
        return merged
    
    def filter_highlights(
        self, 
        highlights: List[Dict], 
        min_confidence: float = 0.3,
        max_highlights: Optional[int] = None
    ) -> List[Dict]:
        """Filter highlights by confidence and optionally limit count"""
        # Filter by confidence
        filtered = [h for h in highlights if h['confidence'] >= min_confidence]
        
        # Sort by confidence (highest first)
        filtered.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Limit count if specified
        if max_highlights and len(filtered) > max_highlights:
            filtered = filtered[:max_highlights]
            logger.info(f"Limited highlights to top {max_highlights} by confidence")
        
        return filtered
    
    def get_analysis_summary(self, highlights: List[Dict]) -> Dict:
        """Generate summary statistics for analysis"""
        if not highlights:
            return {'total_highlights': 0}
        
        event_types = {}
        confidences = []
        durations = []
        
        for highlight in highlights:
            event_type = highlight['event_type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
            confidences.append(highlight['confidence'])
            durations.append(highlight['duration'])
        
        return {
            'total_highlights': len(highlights),
            'event_types': event_types,
            'avg_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'min_confidence': np.min(confidences),
            'avg_duration': np.mean(durations),
            'total_highlight_duration': sum(durations),
            'coverage_percentage': (sum(durations) / self.video_info['duration']) * 100
        }
    
    def save_analysis_report(self, highlights: List[Dict], output_path: str):
        """Save detailed analysis report"""
        report = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'video_path': str(self.video_path),
                'sport_type': self.sport_type,
                'analyzer_version': '1.0.0'
            },
            'video_info': self.video_info,
            'sport_config': self.sport_config,
            'summary': self.get_analysis_summary(highlights),
            'highlights': highlights,
            'analysis_history': self.analysis_history
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Analysis report saved:
