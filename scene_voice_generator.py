import shutil
# from TTS.api import TTS
from pathlib import Path

# Define the configuration values directly
SCENES_DIR = Path("C:/Users/Avinoam Nukrai/Desktop/ml_trailers_project/NLD_gpt_scenes")  # Replace with the path to your scenes directory
model_id = 'tts_models/multilingual/multi-dataset/xtts_v2'
device = 'cpu'
reference_voice_path = 'voices/sample_voice.wav'
tts_language = 'en'
n_audios = 1

def generate_voice(model: TTS, text: str, audio_path: str, reference_voice_path: str, language: str) -> None:
    """Generate voice for the given text and save it to the specified audio path."""
    model.tts_to_file(
        text,
        speaker_wav=reference_voice_path,
        language=language,
        file_path=audio_path,
    )

def generate_voices(model: TTS, n_audios: int, reference_voice_path: str, language: str) -> None:
    """Generate voice for each subplot in the scenes directory."""
    for idx, scene_dir in enumerate(SCENES_DIR.iterdir()):
        if scene_dir.is_dir():
            scene_plot = (scene_dir / "subplot.txt").read_text()
            audio_dir = scene_dir / "audios"
            logger.info(f'Generating audio for scene {idx+1} with plot "{scene_plot}"')

            if audio_dir.exists():
                shutil.rmtree(audio_dir)

            audio_dir.mkdir(parents=True, exist_ok=True)

            for idx in range(n_audios):
                logger.info(f"Generating audio {idx+1}")
                voice_path = audio_dir / f"audio_{idx+1}.wav"
                generate_voice(
                    model, scene_plot, str(voice_path), reference_voice_path, language
                )

def generate():
    tts = TTS(model_name=model_id).to(device)
    generate_voices(
        tts,
        n_audios,
        reference_voice_path,
        tts_language,
    )
