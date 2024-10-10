#!/usr/bin/env python3

import whisperx
from whisperx.utils import get_writer
import torch
import json
import os


DIARIZATIOIN_MODEL = "Revai/reverb-diarization-v2"
# DIARIZATIOIN_MODEL = "pyannote/speaker-diarization-3.1"
OUTPUT_DIR = "output"

def transcribe(input_file, whisper_model="large-v2", num_speakers=2):

    try:
        # Set up device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"

        # Load WhisperX model
        model = whisperx.load_model(whisper_model, device, compute_type=compute_type)

        # Transcribe audio
        audio = whisperx.load_audio(input_file)
        result = model.transcribe(audio, batch_size=16)

        # Load alignment model and align
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # Diarize
        diarize_model = whisperx.DiarizationPipeline(model_name=DIARIZATIOIN_MODEL, use_auth_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"), device=device)
        diarization_segments = diarize_model(audio, num_speakers=num_speakers)

        # Assign speaker labels
        diarize_result = whisperx.assign_word_speakers(diarization_segments, aligned_result)
        diarize_result["language"] = result["language"]

        srt_writer = get_writer("srt", OUTPUT_DIR)
        srt_writer(
            diarize_result,
            input_file,
            {"max_line_width": None, "max_line_count": None, "highlight_words": False}
        )

        with open(OUTPUT_DIR + '/output.json', 'w') as f:
            json.dump(result, f, indent=4)

        print(f"Transcription completed. Output saved to {OUTPUT_DIR}")

        return result
    except Exception as e:
        print(f"An error occurred during transcription: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    input_file = "audio/alt-office/2024-08-14T17:41:46_+19562523979.wav"
    whisper_model = "large-v2"
    num_speakers = 2

    result = transcribe(input_file, whisper_model, num_speakers)
