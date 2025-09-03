"""
Logging utilities for sports highlight generator
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import os

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
    include_module: bool = True
) -> logging.Logger:
    """
    Setup logging configuration for the application
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Custom format string
        include_timestamp: Whether to include timestamp in logs
        include_module: Whether to include module name in logs
    
    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create format string if not provided
    if format_string is None:
        format_parts = []
        
        if include_timestamp:
            format_parts.append("%(asctime)s")
        
        if include_module:
            format_parts.append("%(name)s")
        
        format_parts.extend(["%(levelname)s", "%(message)s"])
        format_string = " - ".join(format_parts)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[]
    )
    
    # Get root logger
    logger = logging.getLogger()
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance for specific module
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class ProgressLogger:
    """Logger with progress tracking capabilities"""
    
    def __init__(self, logger: logging.Logger, total_items: int, log_interval: int = 10):
        """
        Initialize progress logger
        
        Args:
            logger: Logger instance to use
            total_items: Total number of items to process
            log_interval: Log progress every N percent
        """
        self.logger = logger
        self.total_items = total_items
        self.log_interval = log_interval
        self.current_item = 0
        self.last_logged_percent = -1
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1, message: str = "Processing"):
        """
        Update progress and log if needed
        
        Args:
            increment: Number of items processed
            message: Progress message
        """
        self.current_item += increment
        percent = int((self.current_item / self.total_items) * 100)
        
        if percent >= self.last_logged_percent + self.log_interval:
            elapsed = datetime.now() - self.start_time
            rate = self.current_item / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
            
            if rate > 0:
                eta = (self.total_items - self.current_item) / rate
                eta_str = f", ETA: {eta:.1f}s"
            else:
                eta_str = ""
            
            self.logger.info(f"{message}: {percent}% ({self.current_item}/{self.total_items}){eta_str}")
            self.last_logged_percent = percent
    
    def finish(self, message: str = "Completed"):
        """Log completion message"""
        elapsed = datetime.now() - self.start_time
        rate = self.total_items / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        
        self.logger.info(f"{message}: {self.total_items} items in {elapsed.total_seconds():.1f}s "
                        f"({rate:.1f} items/s)")

class TimedLogger:
    """Logger with timing capabilities"""
    
    def __init__(self, logger: logging.Logger, operation_name: str):
        """
        Initialize timed logger
        
        Args:
            logger: Logger instance
            operation_name: Name of operation being timed
        """
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def start(self):
        """Start timing"""
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation_name}...")
    
    def checkpoint(self, message: str):
        """Log checkpoint with elapsed time"""
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            self.logger.info(f"{self.operation_name} - {message} (elapsed: {elapsed.total_seconds():.1f}s)")
    
    def finish(self, success: bool = True):
        """Log completion with total time"""
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            status = "completed successfully" if success else "failed"
            self.logger.info(f"{self.operation_name} {status} in {elapsed.total_seconds():.1f}s")

def log_function_call(func):
    """
    Decorator to log function calls with timing
    
    Usage:
        @log_function_call
        def my_function():
            pass
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        func_name = func.__name__
        
        # Log function start
        logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            elapsed = datetime.now() - start_time
            logger.debug(f"{func_name} completed in {elapsed.total_seconds():.3f}s")
            return result
            
        except Exception as e:
            elapsed = datetime.now() - start_time
            logger.error(f"{func_name} failed after {elapsed.total_seconds():.3f}s: {e}")
            raise
    
    return wrapper

def log_memory_usage(logger: logging.Logger, message: str = "Memory usage"):
    """
    Log current memory usage
    
    Args:
        logger: Logger instance
        message: Log message
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.debug(f"{message}: {memory_mb:.1f} MB")
        
    except ImportError:
        logger.debug("psutil not available for memory monitoring")
    except Exception as e:
        logger.debug(f"Could not get memory info: {e}")

class FileRotatingHandler(logging.Handler):
    """Custom handler that rotates log files based on size"""
    
    def __init__(self, filename: str, max_bytes: int = 10*1024*1024, backup_count: int = 5):
        """
        Initialize rotating file handler
        
        Args:
            filename: Base log filename
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
        """
        super().__init__()
        self.filename = filename
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        # Create directory if needed
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    def emit(self, record):
        """Emit log record, rotating file if needed"""
        try:
            if os.path.exists(self.filename) and os.path.getsize(self.filename) >= self.max_bytes:
                self._rotate_files()
            
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(self.format(record) + '\n')
                
        except Exception:
            self.handleError(record)
    
    def _rotate_files(self):
        """Rotate log files"""
        # Move existing backup files
        for i in range(self.backup_count - 1, 0, -1):
            old_name = f"{self.filename}.{i}"
            new_name = f"{self.filename}.{i + 1}"
            
            if os.path.exists(old_name):
                if os.path.exists(new_name):
                    os.remove(new_name)
                os.rename(old_name, new_name)
        
        # Move current log to .1
        if os.path.exists(self.filename):
            backup_name = f"{self.filename}.1"
            if os.path.exists(backup_name):
                os.remove(backup_name)
            os.rename(self.filename, backup_name)

def setup_production_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Setup production-ready logging configuration
    
    Args:
        log_dir: Directory for log files
    
    Returns:
        Configured logger
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler (WARNING and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_format = logging.Formatter(
        "%(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # Application log file (INFO and above)
    app_log_file = log_path / "sports_highlights.log"
    app_handler = FileRotatingHandler(str(app_log_file), max_bytes=10*1024*1024, backup_count=5)
    app_handler.setLevel(logging.INFO)
    app_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    app_handler.setFormatter(app_format)
    logger.addHandler(app_handler)
    
    # Error log file (ERROR and above)
    error_log_file = log_path / "errors.log"
    error_handler = FileRotatingHandler(str(error_log_file), max_bytes=5*1024*1024, backup_count=3)
    error_handler.setLevel(logging.ERROR)
    error_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    error_handler.setFormatter(error_format)
    logger.addHandler(error_handler)
    
    return logger

def create_session_logger(session_id: str, base_dir: str = "logs") -> logging.Logger:
    """
    Create logger for specific processing session
    
    Args:
        session_id: Unique session identifier
        base_dir: Base directory for logs
    
    Returns:
        Session-specific logger
    """
    session_dir = Path(base_dir) / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    logger_name = f"session.{session_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Session log file
    log_file = session_dir / "session.log"
    handler = logging.FileHandler(str(log_file))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Don't propagate to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger
