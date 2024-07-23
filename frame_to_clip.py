# import logging
import math
import shutil

import librosa
from moviepy.editor import VideoFileClip
from pathlib import Path
from moviepy.editor import concatenate_videoclips, fadein, fadeout


# from common import SCENES_DIR, configs
SCENES_DIR = Path("C:/Users/Avinoam Nukrai/Desktop/ml_trailers_project/NLD_gpt_scenes")  # Replace with the path to your scenes directory

def get_clip(video: VideoFileClip, min_clip_len: int) -> None:
    """Create video clips based on individual frames

    Args:
        video (VideoFileClip): Video file source for the clips
        min_clip_len (int): Minimum clip length
    """
    fps = video.fps

    for idx, scene_dir in enumerate(SCENES_DIR.iterdir()):
        # logger.info(f"Generating clips for scene {idx+1}")
        clip_dir = scene_dir / "clips"
        # audio_filepaths = scene_dir.glob("audios/*.wav")
        frame_paths = scene_dir.glob("frames/*.jpg")

        if clip_dir.exists():
            shutil.rmtree(clip_dir)

        clip_dir.mkdir(parents=True, exist_ok=True)

        # for audio_filepath in audio_filepaths:
        #     audio_filename = audio_filepath.stem
        #     audio_duration = math.ceil(librosa.get_duration(path=audio_filepath))
        #     audio_duration = max(min_clip_len, audio_duration)

        for frame_path in frame_paths:
            frame = int(frame_path.stem.split("_")[-1])

            clip_start = frame // fps
            clip_end = min(clip_start + 19, video.duration)

            clip = video.subclip(clip_start, clip_end)

            clip.write_videofile(
                # f"{clip_dir}/clip_{frame}_{audio_filename}.mp4",
                f"{clip_dir}/clip_{frame}_.mp4",
                verbose=False,
                logger=None,
            )
            # clip.close() # Sometimes the clip is closed before it finished writing

