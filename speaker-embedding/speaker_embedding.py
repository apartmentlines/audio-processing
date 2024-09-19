#!/usr/bin/env python3

import sys
import torch
from pyannote.audio import Inference
from pydub import AudioSegment
from pydub.silence import split_on_silence
import tempfile
import os
import logging
import numpy as np
import scipy.stats

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('speaker_embedding')
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.propagate = False

def remove_silence(audio_path, min_silence_len=1000, silence_thresh=-40):
    logger.debug(f"Removing silence from {audio_path}")
    try:
        audio = AudioSegment.from_file(audio_path)
        logger.debug(f"Audio file loaded, duration: {len(audio)/1000:.2f} seconds")
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        logger.debug(f"Audio split into {len(chunks)} non-silent chunks")
        return sum(chunks)
    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        raise RuntimeError(f"Error processing audio file: {e}")

def extract_speaker_embedding(audio_path):
    logger.debug(f"Extracting speaker embedding from {audio_path}")
    try:
        audio_without_silence = remove_silence(audio_path)
        logger.debug(f"Silence removed, new duration: {len(audio_without_silence)/1000:.2f} seconds")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
            logger.debug(f"Created temporary file: {temp_filename}")
            audio_without_silence.export(temp_filename, format="wav")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Using device: {device}")

        inference = Inference("pyannote/embedding", device=device)
        logger.debug("Inference model loaded")

        embedding = inference(temp_filename)
        logger.debug("Speaker embedding extracted")

        os.unlink(temp_filename)
        logger.debug(f"Temporary file {temp_filename} deleted")

        return embedding
    except Exception as e:
        logger.error(f"Error extracting speaker embedding: {e}")
        raise RuntimeError(f"Error extracting speaker embedding: {e}")

def summarize_embedding(embedding):
    if hasattr(embedding, 'data'):
        embedding = embedding.data

    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)

    embedding_flat = embedding.flatten()

    return {
        "shape": embedding.shape,
        "mean": np.mean(embedding_flat),
        "std": np.std(embedding_flat),
        "min": np.min(embedding_flat),
        "max": np.max(embedding_flat),
        "median": np.median(embedding_flat),
        "skewness": scipy.stats.skew(embedding_flat),
        "kurtosis": scipy.stats.kurtosis(embedding_flat),
        "percentiles": np.percentile(embedding_flat, [25, 50, 75]).tolist()
    }

if __name__ == '__main__':
    if len(sys.argv) != 2:
        logger.error("Incorrect number of arguments")
        print("Usage: python speaker_embedding.py <audio_file_path>")
        sys.exit(1)

    audio_path = sys.argv[1]
    logger.info(f"Processing audio file: {audio_path}")
    try:
        embedding = extract_speaker_embedding(audio_path)
        logger.info("Speaker embedding extracted successfully")
        summary = summarize_embedding(embedding)
        print("Embedding summary:")
        print(f"  Shape: {summary['shape']}")
        print(f"  Mean: {summary['mean']:.4f}")
        print(f"  Std Dev: {summary['std']:.4f}")
        print(f"  Min: {summary['min']:.4f}")
        print(f"  Max: {summary['max']:.4f}")
        print(f"  Median: {summary['median']:.4f}")
        print(f"  Skewness: {summary['skewness']:.4f}")
        print(f"  Kurtosis: {summary['kurtosis']:.4f}")
        print(f"  Percentiles: {summary['percentiles']}")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)
