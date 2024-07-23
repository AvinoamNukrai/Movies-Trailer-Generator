import itertools
# import logging
import shutil
from pathlib import Path

from moviepy.editor import VideoFileClip, concatenate_videoclips

# from common import SCENES_DIR, TRAILER_DIR


def join_clips(clip_combinations: list[tuple[str]], trailer_dir: Path) -> None:
    """Join audio clips to create a trailer.

    Args:
        clip_combinations (list[list[str]]): List of audio clips to be combined
        trailer_dir (Path): Directory save the trailers
    """
    for idx, clip_combination in enumerate(clip_combinations):
        # logger.info(f"Generating trailer {idx+1}")
        trailer_path = trailer_dir / f"trailer_{idx+1}.mp4"
        clips = [VideoFileClip(str(clip_path)) for clip_path in clip_combination]
        trailer = concatenate_videoclips(clips)
        trailer.write_videofile(str(trailer_path))




# import itertools
# import shutil
# from pathlib import Path
# from moviepy.editor import VideoFileClip, concatenate_videoclips
# from moviepy.video.fx import fadein, fadeout
#
# # Define global constants
# TRAILER_DIR = Path(
#     "C:/Users/Avinoam Nukrai/Desktop/ml_trailers_project/trailer")  # Replace with your actual path
# SCENES_DIR = Path(
#     "C:/Users/Avinoam Nukrai/Desktop/ml_trailers_project/NLD_gpt_scenes")  # Replace with your actual path
#
#
# def join_clips(clip_combinations: list[list[str]], trailer_dir: Path) -> None:
#     """Join video clips to create a single trailer.
#
#     Args:
#         clip_combinations (list[list[str]]): List of video clips to be combined
#         trailer_dir (Path): Directory to save the final trailer
#     """
#     # Flatten the list of clip combinations into a single list of clip paths
#     all_clips = list(itertools.chain.from_iterable(clip_combinations))
#
#     # Combine the list of clip paths into VideoFileClip objects
#     clips = [VideoFileClip(str(clip_path)) for clip_path in all_clips]
#
#     # Add fade-in and fade-out effects for smooth transitions
#     clips_with_transitions = []
#     for i, clip in enumerate(clips):
#         if i > 0:
#             # Apply fade-in to all clips after the first
#             clip = fadein(clip, 1)  # 1-second fade-in
#         if i < len(clips) - 1:
#             # Apply fade-out to all clips except the last
#             clip = fadeout(clip, 1)  # 1-second fade-out
#         clips_with_transitions.append(clip)
#
#     # Concatenate all clips into one trailer
#     trailer = concatenate_videoclips(clips_with_transitions, method="compose")
#     trailer_path = trailer_dir / "final_trailer.mp4"  # Single final video file
#     trailer.write_videofile(str(trailer_path), codec="libx264",
#                             audio_codec="aac")
#
#     # Optional: Clean up the directory if needed
#     if TRAILER_DIR.exists():
#         shutil.rmtree(TRAILER_DIR)
#     TRAILER_DIR.mkdir(parents=True, exist_ok=True)