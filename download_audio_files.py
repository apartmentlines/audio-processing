#!/usr/bin/env python3
"""
A script to download and process audio files from S3 based on records from an SQLite database.
"""

import argparse
import logging
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CustomerRecording:
    id: int
    master_id: int
    filename: str
    timestamp: int

class AudioProcessor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.conn = None
        self.setup_logging()

    def setup_logging(self):
        level = logging.DEBUG if self.args.debug else logging.INFO
        logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
        if self.args.debug:
            logging.debug(f"Debug mode enabled. Arguments: {vars(self.args)}")

    def get_db_connection(self):
        try:
            self.conn = sqlite3.connect(self.args.db_name)
            self.conn.row_factory = sqlite3.Row
            logging.debug(f"Successfully connected to the database: {self.args.db_name}")
        except sqlite3.Error as e:
            logging.error(f"Failed to connect to the database: {e}")
            sys.exit(1)

    def fetch_recordings(self) -> List[CustomerRecording]:
        query = """
            SELECT id, master_id, filename, timestamp
            FROM customer_recordings
            LIMIT ? OFFSET ?
        """
        offset = 0
        recordings = []

        try:
            while True:
                cursor = self.conn.cursor()
                cursor.execute(query, (self.args.batch_size, offset))
                batch = cursor.fetchall()

                if not batch:
                    break

                recordings.extend([
                    CustomerRecording(
                        id=row['id'],
                        master_id=row['master_id'],
                        filename=row['filename'],
                        timestamp=row['timestamp']
                    ) for row in batch
                ])

                offset += self.args.batch_size

                if self.args.limit and len(recordings) >= self.args.limit:
                    recordings = recordings[:self.args.limit]
                    break

            logging.info(f"Fetched {len(recordings)} recordings from the database.")
            return recordings
        except sqlite3.Error as e:
            logging.error(f"Failed to fetch recordings: {e}")
            sys.exit(1)

    def process_recordings(self, recordings: List[CustomerRecording]):
        for recording in recordings:
            s3_key = f"{recording.master_id}/{recording.filename}"
            local_path = os.path.join(str(recording.master_id), recording.filename)

            if not self.args.force and os.path.exists(local_path):
                logging.debug(f"Skipping existing file: {local_path}")
                continue

            if self.download_file(s3_key, local_path):
                self.process_audio(local_path)

    def download_file(self, s3_key: str, local_path: str) -> bool:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_path = f"s3://{self.args.bucket}/{s3_key}"
        command = [
            "s3cmd",
            "--config",
            self.args.s3cfg,
            "get",
            s3_path,
            local_path
        ]
        try:
            logging.debug(f"Downloading {s3_path} to {local_path}")
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logging.debug(f"s3cmd stdout: {result.stdout}")
            logging.debug(f"s3cmd stderr: {result.stderr}")
            logging.info(f"Successfully downloaded: {local_path}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to download {s3_key}: {e.stderr}")
            return False
        except FileNotFoundError:
            logging.error("s3cmd is not installed or not found in PATH.")
            sys.exit(1)

    def process_audio(self, file_path: str):
        output_path = f"{file_path}.processed.wav"
        sox_command = [
            "sox", file_path, output_path,
            "rate", "16k",
            "norm",
            "highpass", "100",
            "compand", "0.02,0.20", "5:-60,-40,-10", "-5", "-90", "0.1"
        ]
        try:
            logging.debug(f"Processing audio: {' '.join(sox_command)}")
            subprocess.run(sox_command, check=True, capture_output=True, text=True)
            logging.info(f"Successfully processed: {output_path}")
            os.replace(output_path, file_path)  # Replace original with processed file
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to process {file_path}: {e.stderr}")
        except FileNotFoundError:
            logging.error("Sox is not installed or not found in PATH.")

    def run(self):
        start_time = time.time()
        logging.info("Starting audio processing script")

        self.get_db_connection()
        recordings = self.fetch_recordings()
        self.process_recordings(recordings)

        if self.conn:
            self.conn.close()
            logging.debug("Database connection closed.")

        end_time = time.time()
        logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")
        logging.info("Audio processing script completed")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and process audio files from S3 based on SQLite records."
    )
    parser.add_argument(
        "--bucket", required=True, help="S3 bucket name."
    )
    parser.add_argument(
        "--s3cfg", default=".s3cfg", help="Path to the .s3cfg configuration file (default: .s3cfg)."
    )
    parser.add_argument(
        "--db-name", default="customer_recordings.db", help="SQLite database name (default: customer_recordings.db)."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--force", action="store_true", help="Force download and processing of existing files."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit the total number of recordings to process."
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Number of records to fetch in each database query (default: 100)."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    processor = AudioProcessor(args)
    processor.run()

if __name__ == "__main__":
    main()
