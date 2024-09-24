#!/usr/bin/env python3

# Import required libraries
import os
from pyannote.audio import Pipeline
from pyannote.database import FileFinder
from pyannote.database import registry
import torch

# Set up the database configuration
registry.load_database("database.yml")

hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# Initialize the pre-trained pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)

# Get the protocol for your dataset
protocol = registry.get_protocol("Protocols.SpeakerDiarization.MyProtocol", preprocessors={"audio": FileFinder()})

# Fine-tune the pipeline
pipeline.instantiate({"segmentation": "SegmentationTask", "embedding": "EmbeddingTask"})

pipeline.fit(protocol.train())

# Save the fine-tuned model
pipeline.save_pretrained("/path/to/save/fine_tuned_model")

# Test the fine-tuned model
test_file = {"audio": "/path/to/test/audio.wav"}
diarization = pipeline(test_file)

# Print the diarization result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
