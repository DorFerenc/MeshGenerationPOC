from threading import Lock

class ProgressReporter:
    """
    Handles progress reporting for the reconstruction process.

    This class provides thread-safe methods to update and retrieve the current
    progress of the reconstruction process.
    """

    def __init__(self):
        """Initialize the ProgressReporter with default values."""
        self._progress = 0
        self._status = "Initializing"
        self._lock = Lock()

    def update(self, progress, status):
        """
        Update the current progress and status.

        Args:
            progress (int): The current progress as a percentage (0-100).
            status (str): A brief description of the current status.
        """
        with self._lock:
            self._progress = progress
            self._status = status

    def get_progress(self):
        """
        Get the current progress and status.

        Returns:
            dict: A dictionary containing the current progress and status.
        """
        with self._lock:
            return {
                "progress": self._progress,
                "status": self._status
            }

class StageProgressReporter:
    """
    Handles progress reporting for individual stages of the reconstruction process.

    This class allows for more granular progress reporting within each stage
    of the reconstruction process.
    """

    def __init__(self, main_reporter, stage_name, start_progress, end_progress):
        """
        Initialize the StageProgressReporter.

        Args:
            main_reporter (ProgressReporter): The main progress reporter.
            stage_name (str): The name of the current stage.
            start_progress (int): The starting progress value for this stage.
            end_progress (int): The ending progress value for this stage.
        """
        self.main_reporter = main_reporter
        self.stage_name = stage_name
        self.start_progress = start_progress
        self.end_progress = end_progress
        self.stage_range = end_progress - start_progress

    def update(self, stage_progress, status):
        """
        Update the progress for the current stage.

        Args:
            stage_progress (float): The progress within the current stage (0.0 to 1.0).
            status (str): A brief description of the current status.
        """
        overall_progress = int(self.start_progress + self.stage_range * stage_progress)
        self.main_reporter.update(overall_progress, f"{self.stage_name}: {status}")

def create_stage_reporters(main_reporter):
    """
    Create stage progress reporters for each stage of the reconstruction process.

    Args:
        main_reporter (ProgressReporter): The main progress reporter.

    Returns:
        dict: A dictionary of StageProgressReporters for each stage.
    """
    return {
        "input_processing": StageProgressReporter(main_reporter, "Input Processing", 0, 20),
        "mesh_generation": StageProgressReporter(main_reporter, "Mesh Generation", 20, 50),
        "texturing": StageProgressReporter(main_reporter, "Texturing", 50, 80),
        "obj_conversion": StageProgressReporter(main_reporter, "OBJ Conversion", 80, 100)
    }
