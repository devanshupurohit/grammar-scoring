from pathlib import Path
import pandas as pd
import torch
from faster_whisper import WhisperModel

from .config import ASR_MODEL_SIZE


def load_whisper_model(model_size=ASR_MODEL_SIZE):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"Loading Whisper model: {model_size}")
    print(f"Device: {device}")
    print(f"Compute type: {compute_type}")

    return WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )


def transcribe_audio(audio_path, model):
    segments, _ = model.transcribe(
        str(audio_path),
        language="en",
        beam_size=5,
        vad_filter=True,
    )

    return " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    ).strip()


def transcribe_split(
    dataframe,
    split,
    audio_dir,
    output_file,
    model=None,
    resume=True,
):
    output_file = Path(output_file)

    existing = {}
    if resume and output_file.exists():
        cached = pd.read_csv(output_file)
        if "filename" in cached.columns and "transcript" in cached.columns:
            existing = dict(
                zip(
                    cached["filename"].astype(str),
                    cached["transcript"].fillna("").astype(str),
                )
            )

    if model is None:
        model = load_whisper_model()

    rows = []

    for i, filename in enumerate(dataframe["filename"].astype(str), start=1):
        if filename in existing:
            transcript = existing[filename]
        else:
            audio_path = Path(audio_dir) / Path(filename).name
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Audio file not found: {audio_path}"
                )
            transcript = transcribe_audio(audio_path, model)

        rows.append({
            "filename": filename,
            "transcript": transcript,
        })

        if i % 10 == 0 or i == len(dataframe):
            print(f"{split}: {i}/{len(dataframe)}")

        # Save incrementally so an interrupted run can resume.
        pd.DataFrame(rows).to_csv(output_file, index=False)

    return pd.DataFrame(rows)
