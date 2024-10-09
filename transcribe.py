#/usr/bin/env python3

import whisperx
import torch
from pyannote.audio import Pipeline
import os

def transcribe(input_file, whisper_model="large-v2", language="en", min_speakers=2, max_speakers=2):
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    # Load WhisperX model
    model = whisperx.load_model(whisper_model, device, compute_type=compute_type, language=language)

    # Transcribe audio
    audio = whisperx.load_audio(input_file)
    result = model.transcribe(audio, batch_size=16)

    # Load alignment model and align
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # Load custom diarization pipeline
    diarization_pipeline = Pipeline.from_pretrained(
        "Revai/reverb-diarization-v2",
        use_auth_token=os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    )

    # Perform diarization
    diarization = diarization_pipeline(input_file, min_speakers=min_speakers, max_speakers=max_speakers)

    # Assign speaker labels
    result = whisperx.assign_word_speakers(diarization, result)

    # Write output to SRT file
    output_file = os.path.splitext(input_file)[0] + ".srt"
    whisperx.write_srt(result["segments"], output_file)

    print(f"Transcription completed. Output saved to {output_file}")

    return result

# Example usage
if __name__ == "__main__":
    input_file = "path/to/your/audio/file.wav"
    whisper_model = "large-v2"
    language = "en"
    min_speakers = 2
    max_speakers = 2

    result = transcribe(input_file, whisper_model, language, min_speakers, max_speakers)
