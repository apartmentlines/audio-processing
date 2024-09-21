#!/usr/bin/env python3

"""
A script to generate UEM files for audio data, which are needed for training
a speaker diarization model for pyannote.audio.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import subprocess


class UEMGenerator:
    def __init__(
        self,
        data_dir: Path,
        uem_dir: Path,
        list_shorter_than: Optional[float] = None,
        list_longer_than: Optional[float] = None,
        total: bool = False,
        debug: bool = False,
    ):
        self.data_dir = data_dir
        self.uem_dir = uem_dir
        self.list_shorter_than = list_shorter_than
        self.list_longer_than = list_longer_than
        self.total = total
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

    def calculate_total_duration(self) -> tuple[str, int]:
        total_seconds = 0
        total_count = 0
        for uem_file in self.uem_dir.glob("*.uem"):
            with uem_file.open("r") as f:
                for line in f:
                    _, _, start, end = line.strip().split()
                    duration = float(end) - float(start)
                    total_seconds += duration
                    total_count += 1

        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}", total_count

    def list_files_by_duration(self, threshold: float, shorter: bool = True):
        filtered_files = []
        comparison = (lambda x, y: x < y) if shorter else (lambda x, y: x > y)
        comparison_text = "shorter" if shorter else "longer"

        for uem_file in self.uem_dir.glob("*.uem"):
            with uem_file.open("r") as f:
                for line in f:
                    _, _, start, end = line.strip().split()
                    duration = float(end) - float(start)
                    if comparison(duration, threshold):
                        wav_file = uem_file.stem + ".wav"
                        logging.warning(f"File {wav_file} is {comparison_text} than {threshold} seconds: {duration:.2f} seconds")
                        filtered_files.append(wav_file)

        if filtered_files:
            print(f"Files {comparison_text} than the specified threshold:")
            for file_name in filtered_files:
                print(file_name)
        else:
            print(f"No files {comparison_text} than {threshold} seconds found.")

    def list_shorter_files(self, threshold: float):
        self.list_files_by_duration(threshold, shorter=True)

    def list_longer_files(self, threshold: float):
        self.list_files_by_duration(threshold, shorter=False)

    def run(self):
        if self.total:
            total_duration, total_count = self.calculate_total_duration()
            logging.info(f"Total duration of {total_count} files: {total_duration}")
        elif self.list_shorter_than is not None:
            self.list_shorter_files(self.list_shorter_than)
        elif self.list_longer_than is not None:
            self.list_longer_files(self.list_longer_than)
        else:
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
    parser.add_argument(
        "--list-shorter-than",
        type=float,
        help="List UEM files shorter than the specified number of seconds.",
    )
    parser.add_argument(
        "--list-longer-than",
        type=float,
        help="List UEM files longer than the specified number of seconds.",
    )
    parser.add_argument("--total", action="store_true", help="Calculate total duration of all UEM files.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    try:
        generator = UEMGenerator(
            data_dir=args.data_dir,
            uem_dir=args.uem_dir,
            list_shorter_than=args.list_shorter_than,
            list_longer_than=args.list_longer_than,
            total=args.total,
            debug=args.debug,
        )
        generator.run()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
