# Copied from eva_p1_config_gpt.py
class AnalysisConfig:
    """Configuration class for video analysis parameters."""
    
    # Basic parameters
    frame_skip: int = 5
    max_frames: int = 300
    face_confidence_threshold: float = 0.7

    # Enhanced detection parameters
    anatomical_threshold: float = 0.6
    face_quality_threshold: float = 0.8
    artifact_threshold: float = 0.5

    # Analysis modes
    enable_temporal_analysis: bool = True
    enable_frequency_analysis: bool = True
    enable_optical_flow: bool = True
    enable_face_landmarks: bool = True
    enable_eye_blink_analysis: bool = True

    # Performance parameters
    batch_size: int = 32
    num_workers: int = 4
    gpu_enabled: bool = True

    # Logging parameters
    log_level: str = "INFO"
    save_detailed_logs: bool = True
    log_training_process: bool = True

    # Output parameters
    output_format: str = "json"
    save_frames: bool = False
    create_summary: bool = True
    
    def __post_init__(self):
        """Валідація параметрів після ініціалізації."""
        if self.frame_skip < 1:
            self.frame_skip = 1
        if self.max_frames < 10:
            self.max_frames = 10
        if not 0.0 <= self.face_confidence_threshold <= 1.0:
            self.face_confidence_threshold = 0.7

