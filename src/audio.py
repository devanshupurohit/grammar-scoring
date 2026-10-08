from pathlib import Path
import soundfile as sf


def get_audio_path(filename, split, data_dir):
    filename = Path(str(filename)).name

    if split == "train":
        return Path(data_dir) / "train" / filename
    if split == "test":
        return Path(data_dir) / "test" / filename

    raise ValueError(f"Unknown split: {split}")


def get_audio_metadata(audio_path):
    info = sf.info(str(audio_path))
    return {
        "sample_rate": info.samplerate,
        "duration": info.duration,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
    }
