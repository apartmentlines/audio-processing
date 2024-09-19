#!/usr/bin/env python3

"""
A script to generate UEM files for audio data, which are needed for training
a speaker diarization model for pyannote.audio.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List
import subprocess


class UEMGenerator:
    def __init__(
        self,
        data_dir: Path,
        uem_dir: Path,
        debug: bool = False,
    ):
        self.data_dir = data_dir
        self.uem_dir = uem_dir
        self.debug = debug
        self.setup_logging()

    def setup_logging(self):
        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        if self.debug:
            logging.debug(f"Debug mode enabled. Arguments: {self.__dict__}")

    def get_audio_files(self) -> List[Path]:
        audio_files = list(self.data_dir.glob("*.wav"))
        logging.info(f"Found {len(audio_files)} audio WAV files in {self.data_dir}")
        return audio_files

    def get_audio_duration(self, audio_file: Path) -> float:
        try:
            result = subprocess.run(
                ["soxi", "-D", str(audio_file)],
                capture_output=True,
                text=True,
                check=True,
            )
            duration = float(result.stdout.strip())
            logging.debug(f"Duration of {audio_file.name}: {duration} seconds")
            return duration
        except subprocess.CalledProcessError as e:
            logging.error(f"Error getting duration for {audio_file}: {e}")
            return 0.0

    def generate_uem_file(self, audio_file: Path, duration: float):
        uem_file = self.uem_dir / f"{audio_file.stem}.uem"
        try:
            uem_file.parent.mkdir(parents=True, exist_ok=True)
            with uem_file.open("w") as f:
                f.write(f"{audio_file.stem} 1 0.000 {duration:.3f}\n")
            logging.info(f"Generated UEM file: {uem_file}")
        except IOError as e:
            logging.error(f"Error writing UEM file {uem_file}: {e}")

    def process_audio_files(self):
        audio_files = self.get_audio_files()
        for audio_file in audio_files:
            duration = self.get_audio_duration(audio_file)
            if duration > 0:
                self.generate_uem_file(audio_file, duration)

    def run(self):
        logging.info("Starting UEM file generation")
        self.process_audio_files()
        logging.info("UEM file generation completed")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate UEM files for audio data needed for training a speaker diarization model."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("audio"),
        help="Directory containing audio files (default: %(default)s).",
    )
    parser.add_argument(
        "--uem-dir",
        type=Path,
        default=Path("uem"),
        help="Directory for storing UEM files (default: %(default)s).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    try:
        generator = UEMGenerator(
            data_dir=args.data_dir,
            uem_dir=args.uem_dir,
            debug=args.debug,
        )
        generator.run()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
