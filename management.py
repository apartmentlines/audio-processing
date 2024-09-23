#!/usr/bin/env python3

"""
A management script for various audio processing tasks.
"""

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict
import random
import yaml


class AudioManager:
    def __init__(self, db_name: str, uem_dir: Path, audio_dir: Path, diarization_dir: Path, eaf_dir: Path, rttm_dir: Path, debug: bool = False):
        self.db_name = db_name
        self.uem_dir = uem_dir
        self.audio_dir = audio_dir
        self.diarization_dir = diarization_dir
        self.eaf_dir = eaf_dir
        self.rttm_dir = rttm_dir
        self.debug = debug
        self.setup_logging()
        self.conn = self.get_db_connection()

    def setup_logging(self):
        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        if self.debug:
            logging.debug(f"Debug mode enabled. Arguments: {self.__dict__}")

    def calculate_total_duration(self) -> Tuple[str, int]:
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

    def list_files_by_duration(self, threshold: float, shorter: bool = True) -> List[str]:
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

        return filtered_files

    def run(self, args):
        try:
            if args.verify_data:
                self.verify_data()
            elif args.total:
                self.print_total_duration()
            elif args.list_shorter_than is not None:
                self.print_files_by_duration(args.list_shorter_than, shorter=True)
            elif args.list_longer_than is not None:
                self.print_files_by_duration(args.list_longer_than, shorter=False)
            elif args.create_split:
                self.create_pyannote_split(args.split_ratios, args.output_dir)
        finally:
            if self.conn:
                self.conn.close()

    def print_total_duration(self):
        total_duration, total_count = self.calculate_total_duration()
        logging.info(f"Total duration of {total_count} files: {total_duration}")

    def print_files_by_duration(self, threshold: float, shorter: bool):
        filtered_files = self.list_files_by_duration(threshold, shorter=shorter)
        comparison = "shorter" if shorter else "longer"
        if filtered_files:
            print(f"Files {comparison} than {threshold} seconds:")
            for file_name in filtered_files:
                print(file_name)
        else:
            print(f"No files {comparison} than {threshold} seconds found.")

    def create_pyannote_split(self, split_ratios: str, output_dir: Path):
        logging.info(f"Creating pyannote.audio split with ratios {split_ratios}")

        try:
            ratios = [float(r) for r in split_ratios.split(',')]
            if len(ratios) != 3 or sum(ratios) != 1.0:
                raise ValueError("Split ratios must be three comma-separated numbers that sum to 1.0")

            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute("SELECT id, filename FROM customer_recordings WHERE eaf_complete = 1")
                all_recordings = cursor.fetchall()

            if not all_recordings:
                logging.warning("No completed recordings found for creating split.")
                return

            random.shuffle(all_recordings)
            total = len(all_recordings)
            train_end = int(total * ratios[0])
            dev_end = train_end + int(total * ratios[1])

            train_set = all_recordings[:train_end]
            development_set = all_recordings[train_end:dev_end]
            test_set = all_recordings[dev_end:]

            output_dir.mkdir(parents=True, exist_ok=True)

            # Create database.yml
            database = {}
            for rec in all_recordings:
                basename = Path(rec[1]).stem
                database[basename] = {
                    'uri': f'audio/{rec[1]}',
                    'annotation': f'rttm/{basename}.rttm',
                    'annotated': f'uem/{basename}.uem'
                }
            with open(output_dir / 'database.yml', 'w') as f:
                yaml.dump({'Databases': {'MyDatabase': database}}, f)

            # Create split files
            for name, dataset in [('train', train_set), ('development', development_set), ('test', test_set)]:
                with open(output_dir / f'{name}.txt', 'w') as f:
                    for rec in dataset:
                        f.write(f"{rec[1]}\n")

            # Create config.yml
            config = {
                'Protocols': {
                    'MyDatabase': {
                        'SpeakerDiarization': {
                            'MyProtocol': {
                                'train': {'uri': 'train.txt'},
                                'development': {'uri': 'development.txt'},
                                'test': {'uri': 'test.txt'}
                            }
                        }
                    }
                }
            }
            with open(output_dir / 'config.yml', 'w') as f:
                yaml.dump(config, f)

            logging.info(f"Split created: {len(train_set)} training, {len(development_set)} development, {len(test_set)} test samples")
            logging.info(f"Configuration files written to {output_dir}")

        except (sqlite3.Error, ValueError) as e:
            logging.error(f"Error when creating split: {e}")

    def verify_data(self):
        logging.info("Verifying data files...")
        filenames = self.get_filenames_from_db()
        directories = {
            "audio": Path("audio"),
            "diarization-results": Path("diarization-results"),
            "eaf": Path("eaf"),
            "rttm": Path("rttm"),
            "uem": self.uem_dir
        }
        extensions = {
            "audio": ".wav",
            "diarization-results": ".json",
            "eaf": ".eaf",
            "rttm": ".rttm",
            "uem": ".uem"
        }
        for filename in filenames:
            for dir_name, dir_path in directories.items():
                expected_file = dir_path / (filename.stem + extensions[dir_name])
                if expected_file.exists():
                    logging.debug(f"File exists: {expected_file}")
                else:
                    logging.warning(f"File missing: {expected_file}")
        logging.info("Data verification completed.")

    def get_db_connection(self):
        try:
            conn = sqlite3.connect(self.db_name)
            return conn
        except sqlite3.Error as e:
            logging.error(f"Database connection error: {e}")
            return None

    def get_filenames_from_db(self) -> List[Path]:
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT filename FROM customer_recordings")
            filenames = [Path(row[0]) for row in cursor.fetchall()]
            return filenames
        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            return []


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Management script for audio processing tasks."
    )
    parser.add_argument(
        "--db-name",
        default="customer_recordings.db",
        help="SQLite database name (default: %(default)s).",
    )
    parser.add_argument(
        "--uem-dir",
        type=Path,
        default=Path("uem"),
        help="Directory containing UEM files (default: %(default)s).",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("audio"),
        help="Directory containing audio files (default: %(default)s).",
    )
    parser.add_argument(
        "--diarization-dir",
        type=Path,
        default=Path("diarization-results"),
        help="Directory containing diarization results (default: %(default)s).",
    )
    parser.add_argument(
        "--eaf-dir",
        type=Path,
        default=Path("eaf"),
        help="Directory containing EAF files (default: %(default)s).",
    )
    parser.add_argument(
        "--rttm-dir",
        type=Path,
        default=Path("rttm"),
        help="Directory containing RTTM files (default: %(default)s).",
    )
    parser.add_argument(
        "--total",
        action="store_true",
        help="Calculate total duration of all UEM files.",
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
    parser.add_argument(
        "--verify-data",
        action="store_true",
        help="Verify that all necessary files exist.",
    )
    parser.add_argument(
        "--create-split",
        action="store_true",
        help="Create train/development/test split for pyannote.audio speaker diarization.",
    )
    parser.add_argument(
        "--split-ratios",
        type=str,
        default="0.8,0.1,0.1",
        help="Comma-separated ratios for train,development,test split (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pyannote_data"),
        help="Directory to store pyannote.audio configuration files (default: %(default)s).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    manager = AudioManager(
        db_name=args.db_name,
        uem_dir=args.uem_dir,
        audio_dir=args.audio_dir,
        diarization_dir=args.diarization_dir,
        eaf_dir=args.eaf_dir,
        rttm_dir=args.rttm_dir,
        debug=args.debug,
    )
    manager.run(args)


if __name__ == "__main__":
    main()
