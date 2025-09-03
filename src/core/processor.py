"""
Video processing module for extracting and creating highlight clips
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import timedelta
import subprocess
import json

try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
    from moviepy.config import check_and_download_cmd
except ImportError:
    raise ImportError("Please install moviepy: pip install moviepy")

from config.settings import (
    CLIP_PADDING, MAX_CLIP_DURATION, MIN_CLIP_DURATION,
    VIDEO_CODEC, AUDIO_CODEC, DEFAULT_OUTPUT_DIR
)

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles video processing and clip extraction"""
    
    def __init__(
        self,
        video_codec: str = VIDEO_CODEC,
        audio_codec: str = AUDIO_CODEC,
        temp_dir: Optional[str] = None
    ):
        """
        Initialize video processor
        
        Args:
            video_codec: Video codec for output files
            audio_codec: Audio codec for output files
            temp_dir: Temporary directory for processing
        """
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        self.temp_dir = temp_dir or "temp"
        
        # Ensure temp directory exists
        Path(self.temp_dir).mkdir(exist_ok=True)
        
        self.video_clip = None
        self.video_path = None
        
        logger.info(f"VideoProcessor initialized with {video_codec}/{audio_codec}")
    
    def load_video(self, video_path: str) -> bool:
        """
        Load video file for processing
        
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
            
            self.video_clip = VideoFileClip(str(video_path))
            logger.info(f"Video loaded: {video_path} ({self.video_clip.duration:.1f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load video: {e}")
            return False
    
    def extract_clip(
        self,
        start_time: float,
        end_time: float,
        output_path: str,
        add_padding: bool = True,
        quality: str = "medium"
    ) -> bool:
        """
        Extract a single clip from the video
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Output file path
            add_padding: Whether to add padding around the clip
            quality: Output quality (low, medium, high)
        
        Returns:
            Success status
        """
        if not self.video_clip:
            logger.error("No video loaded")
            return False
        
        try:
            # Apply padding if requested
            if add_padding:
                start_time = max(0, start_time - CLIP_PADDING)
                end_time = min(self.video_clip.duration, end_time + CLIP_PADDING)
            
            # Validate duration
            duration = end_time - start_time
            if duration < MIN_CLIP_DURATION:
                logger.warning(f"Clip too short ({duration:.1f}s), skipping")
                return False
            
            if duration > MAX_CLIP_DURATION:
                logger.warning(f"Clip too long ({duration:.1f}s), truncating")
                end_time = start_time + MAX_CLIP_DURATION
            
            # Extract clip
            clip = self.video_clip.subclip(start_time, end_time)
            
            # Set quality parameters
            quality_settings = self._get_quality_settings(quality)
            
            # Write clip
            clip.write_videofile(
                output_path,
                codec=self.video_codec,
                audio_codec=self.audio_codec,
                verbose=False,
                logger=None,
                **quality_settings
            )
            
            clip.close()
            
            logger.info(f"Clip extracted: {output_path} ({duration:.1f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract clip: {e}")
            return False
    
    def extract_highlights(
        self,
        highlights: List[Dict],
        output_dir: str,
        quality: str = "medium",
        progress_callback: Optional[callable] = None
    ) -> List[str]:
        """
        Extract multiple highlight clips
        
        Args:
            highlights: List of highlight events
            output_dir: Output directory
            quality: Output quality
            progress_callback: Optional progress callback
        
        Returns:
            List of successfully created clip paths
        """
        if not self.video_clip:
            logger.error("No video loaded")
            return []
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        successful_clips = []
        
        for i, highlight in enumerate(highlights):
            try:
                # Generate filename
                filename = self._generate_clip_filename(highlight, i + 1)
                clip_path = output_path / filename
                
                # Calculate times
                start_time = highlight['timestamp']
                end_time = start_time + highlight['duration']
                
                # Extract clip
                success = self.extract_clip(
                    start_time, end_time, str(clip_path), quality=quality
                )
                
                if success:
                    successful_clips.append(str(clip_path))
                
                # Progress callback
                if progress_callback:
                    progress = (i + 1) / len(highlights)
                    progress_callback(progress)
                    
            except Exception as e:
                logger.error(f"Failed to process highlight {i + 1}: {e}")
        
        logger.info(f"Extracted {len(successful_clips)}/{len(highlights)} clips")
        return successful_clips
    
    def create_highlight_reel(
        self,
        clip_paths: List[str],
        output_path: str,
        add_transitions: bool = True,
        add_titles: bool = True,
        quality: str = "medium"
    ) -> bool:
        """
        Create a combined highlight reel from individual clips
        
        Args:
            clip_paths: List of clip file paths
            output_path: Output path for combined reel
            add_transitions: Whether to add transitions between clips
            add_titles: Whether to add title cards
            quality: Output quality
        
        Returns:
            Success status
        """
        try:
            # Load all clips
            clips = []
            valid_paths = []
            
            for clip_path in clip_paths:
                if os.path.exists(clip_path):
                    try:
                        clip = VideoFileClip(clip_path)
                        clips.append(clip)
                        valid_paths.append(clip_path)
                    except Exception as e:
                        logger.warning(f"Could not load clip {clip_path}: {e}")
                else:
                    logger.warning(f"Clip not found: {clip_path}")
            
            if not clips:
                logger.error("No valid clips found for highlight reel")
                return False
            
            # Add titles if requested
            if add_titles:
                clips = self._add_title_cards(clips, valid_paths)
            
            # Add transitions if requested
            if add_transitions:
                clips = self._add_transitions(clips)
            
            # Concatenate clips
            final_reel = concatenate_videoclips(clips, method="compose")
            
            # Set quality and write
            quality_settings = self._get_quality_settings(quality)
            final_reel.write_videofile(
                output_path,
                codec=self.video_codec,
                audio_codec=self.audio_codec,
                verbose=False,
                logger=None,
                **quality_settings
            )
            
            # Clean up
            final_reel.close()
            for clip in clips:
                try:
                    clip.close()
                except:
                    pass
            
            logger.info(f"Highlight reel created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create highlight reel: {e}")
            return False
    
    def create_summary_video(
        self,
        highlights: List[Dict],
        output_path: str,
        max_duration: float = 300,  # 5 minutes
        quality: str = "medium"
    ) -> bool:
        """
        Create a summary video with the best highlights
        
        Args:
            highlights: List of highlight events
            output_path: Output path
            max_duration: Maximum duration for summary
            quality: Output quality
        
        Returns:
            Success status
        """
        if not highlights:
            logger.error("No highlights provided")
            return False
        
        # Sort by confidence and select best highlights
        sorted_highlights = sorted(highlights, key=lambda x: x['confidence'], reverse=True)
        
        selected_highlights = []
        total_duration = 0
        
        for highlight in sorted_highlights:
            clip_duration = highlight['duration'] + (CLIP_PADDING * 2)
            if total_duration + clip_duration <= max_duration:
                selected_highlights.append(highlight)
                total_duration += clip_duration
            else:
                break
        
        logger.info(f"Selected {len(selected_highlights)} highlights for summary "
                   f"({total_duration:.1f}s/{max_duration}s)")
        
        # Extract selected clips to temp directory
        temp_dir = Path(self.temp_dir) / "summary_clips"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            clip_paths = self.extract_highlights(
                selected_highlights, 
                str(temp_dir), 
                quality=quality
            )
            
            if not clip_paths:
                logger.error("No clips extracted for summary")
                return False
            
            # Create combined reel
            success = self.create_highlight_reel(
                clip_paths, output_path, add_titles=True, quality=quality
            )
            
            # Clean up temp clips
            self._cleanup_temp_files(temp_dir)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to create summary video: {e}")
            return False
    
    def add_overlay_info(
        self,
        clip_path: str,
        highlight_info: Dict,
        output_path: str
    ) -> bool:
        """
        Add overlay information to a clip
        
        Args:
            clip_path: Input clip path
            highlight_info: Highlight information to overlay
            output_path: Output path with overlay
        
        Returns:
            Success status
        """
        try:
            clip = VideoFileClip(clip_path)
            
            # Create text overlay
            text_content = f"{highlight_info['event_type'].replace('_', ' ').title()}\n"
            text_content += f"Confidence: {highlight_info['confidence']:.2f}\n"
            text_content += f"Time: {self._format_timestamp(highlight_info['timestamp'])}"
            
            text_clip = TextClip(
                text_content,
                fontsize=24,
                color='white',
                font='Arial-Bold',
                stroke_color='black',
                stroke_width=2
            ).set_duration(clip.duration).set_position(('left', 'top')).set_margin(10)
            
            # Composite video
            final_clip = CompositeVideoClip([clip, text_clip])
            
            # Write with overlay
            final_clip.write_videofile(
                output_path,
                codec=self.video_codec,
                audio_codec=self.audio_codec,
                verbose=False,
                logger=None
            )
            
            # Clean up
            final_clip.close()
            text_clip.close()
            clip.close()
            
            logger.info(f"Overlay added: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add overlay: {e}")
            return False
    
    def _generate_clip_filename(self, highlight: Dict, index: int) -> str:
        """Generate filename for highlight clip"""
        event_type = highlight['event_type'].replace(' ', '_').replace('_', '_')
        timestamp_str = self._format_timestamp(highlight['timestamp'], for_filename=True)
        confidence_str = f"{highlight['confidence']:.2f}".replace('.', '')
        
        filename = f"highlight_{index:02d}_{event_type}_{timestamp_str}_c{confidence_str}.mp4"
        return filename
    
    def _format_timestamp(self, seconds: float, for_filename: bool = False) -> str:
        """Format timestamp for display or filename"""
        td = timedelta(seconds=int(seconds))
        if for_filename:
            return f"{int(td.total_seconds()//60):02d}m{int(td.total_seconds()%60):02d}s"
        else:
            return f"{int(td.total_seconds()//3600):02d}:{int((td.total_seconds()%3600)//60):02d}:{int(td.total_seconds()%60):02d}"
    
    def _get_quality_settings(self, quality: str) -> Dict:
        """Get quality settings for video encoding"""
        settings = {
            'low': {
                'bitrate': '500k',
                'fps': 24,
            },
            'medium': {
                'bitrate': '2000k',
                'fps': 30,
            },
            'high': {
                'bitrate': '5000k',
                'fps': 30,
            }
        }
        
        return settings.get(quality, settings['medium'])
    
    def _add_title_cards(self, clips: List, clip_paths: List[str]) -> List:
        """Add title cards to clips"""
        enhanced_clips = []
        
        for i, (clip, path) in enumerate(zip(clips, clip_paths)):
            # Extract highlight info from filename
            filename = os.path.basename(path)
            title = filename.replace('.mp4', '').replace('_', ' ').title()
            
            # Create title card
            title_clip = TextClip(
                f"Highlight {i + 1}: {title}",
                fontsize=30,
                color='white',
                bg_color='black',
                size=clip.size
            ).set_duration(2)  # 2-second title card
            
            enhanced_clips.extend([title_clip, clip])
        
        return enhanced_clips
    
    def _add_transitions(self, clips: List) -> List:
        """Add simple transitions between clips"""
        # For now, just add short fade transitions
        # This is a simplified implementation
        enhanced_clips = []
        
        for i, clip in enumerate(clips):
            if i > 0:  # Add fade-in for all clips except first
                clip = clip.fadein(0.5)
            if i < len(clips) - 1:  # Add fade-out for all clips except last
                clip = clip.fadeout(0.5)
            enhanced_clips.append(clip)
        
        return enhanced_clips
    
    def _cleanup_temp_files(self, temp_dir: Path):
        """Clean up temporary files"""
        try:
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not clean up temp directory: {e}")
    
    def get_video_info(self) -> Dict:
        """Get information about loaded video"""
        if not self.video_clip:
            return {}
        
        return {
            'path': str(self.video_path),
            'duration': self.video_clip.duration,
            'fps': self.video_clip.fps,
            'size': self.video_clip.size,
            'aspect_ratio': self.video_clip.w / self.video_clip.h if self.video_clip.h > 0 else 0
        }
    
    def create_thumbnail(
        self,
        timestamp: float,
        output_path: str,
        size: Tuple[int, int] = (320, 180)
    ) -> bool:
        """
        Create thumbnail image from video at specific timestamp
        
        Args:
            timestamp: Time in seconds to extract thumbnail
            output_path: Output image path
            size: Thumbnail size (width, height)
        
        Returns:
            Success status
        """
        if not self.video_clip:
            logger.error("No video loaded")
            return False
        
        try:
            # Extract frame at timestamp
            frame = self.video_clip.get_frame(timestamp)
            
            # Resize if needed
            if size:
                from PIL import Image
                import numpy as np
                
                image = Image.fromarray(frame)
                image = image.resize(size, Image.Resampling.LANCZOS)
                image.save(output_path)
            else:
                # Save original size
                from PIL import Image
                Image.fromarray(frame).save(output_path)
            
            logger.info(f"Thumbnail created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create thumbnail: {e}")
            return False
    
    def create_highlight_thumbnails(
        self,
        highlights: List[Dict],
        output_dir: str,
        size: Tuple[int, int] = (320, 180)
    ) -> List[str]:
        """Create thumbnails for all highlights"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        thumbnail_paths = []
        
        for i, highlight in enumerate(highlights):
            thumbnail_name = f"thumbnail_{i+1:02d}_{highlight['event_type']}.jpg"
            thumbnail_path = output_path / thumbnail_name
            
            if self.create_thumbnail(
                highlight['timestamp'], 
                str(thumbnail_path), 
                size
            ):
                thumbnail_paths.append(str(thumbnail_path))
        
        return thumbnail_paths
    
    def close(self):
        """Clean up resources"""
        if self.video_clip:
            try:
                self.video_clip.close()
                self.video_clip = None
                logger.debug("Video clip closed")
            except:
                pass
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.close()
