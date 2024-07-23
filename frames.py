# import logging
import shutil
from pathlib import Path
import cv2

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

def create_screenshots(video_path: str, frames_dir: str, n_frames: int) -> None:
    """Take multiple frames from a video file and save them to the specified directory.

    Args:
        video_path (str): Path to the video file.
        frames_dir (str): Directory where the frames will be saved.
        n_frames (int): Number of frames to extract from the video.
    """
    frames_path = Path(frames_dir)

    # If the frames directory exists, remove it and create a new one
    if frames_path.exists():
        shutil.rmtree(frames_path)
    frames_path.mkdir(parents=True, exist_ok=True)

    cam = cv2.VideoCapture(video_path)
    total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    # logger.info(f"Total frames in video: {total_frames}")

    currentframe = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Save frame if it is one of the desired frames
        if currentframe % (total_frames // n_frames) == 0:
            img_path = frames_path / f"frame_{currentframe}.jpg"
            cv2.imwrite(str(img_path), frame)
            # logger.info(f"Saved frame {currentframe} to {img_path}")

        currentframe += 1

    cam.release()
    cv2.destroyAllWindows()
