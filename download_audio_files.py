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
    def __init__(self,
                 bucket: str,
                 s3cfg: str,
                 db_name: str,
                 directory: str,
                 debug: bool = False,
                 force: bool = False,
                 limit: Optional[int] = None,
                 batch_size: int = 100):
        self.bucket = bucket
        self.s3cfg = s3cfg
        self.db_name = db_name
        self.directory = directory
        self.debug = debug
        self.force = force
        self.limit = limit
        self.batch_size = batch_size
        self.conn = None
        self.setup_logging()

    def setup_logging(self):
        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
        if self.debug:
            logging.debug(f"Debug mode enabled. Arguments: {self.__dict__}")

    def get_db_connection(self):
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.conn.row_factory = sqlite3.Row
            logging.debug(f"Successfully connected to the database: {self.db_name}")
            self.create_table_if_not_exists()
        except sqlite3.Error as e:
            logging.error(f"Failed to connect to the database: {e}")
            sys.exit(1)

    def create_table_if_not_exists(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS customer_recordings (
            id INTEGER PRIMARY KEY,
            master_id INTEGER NOT NULL,
            filename VARCHAR(255) NOT NULL,
            timestamp BIGINT NOT NULL DEFAULT 0
        );
        """
        create_index_queries = [
            "CREATE INDEX IF NOT EXISTS idx_master_id ON customer_recordings(master_id);",
            "CREATE INDEX IF NOT EXISTS idx_filename ON customer_recordings(filename);",
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON customer_recordings(timestamp);"
        ]

        try:
            with self.conn:
                self.conn.execute(create_table_query)
                for index_query in create_index_queries:
                    self.conn.execute(index_query)
            logging.debug("customer_recordings table and indexes created or already exist.")
        except sqlite3.Error as e:
            logging.error(f"Failed to create table or indexes: {e}")
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
                cursor.execute(query, (self.batch_size, offset))
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

                offset += self.batch_size

                if self.limit and len(recordings) >= self.limit:
                    recordings = recordings[:self.limit]
                    break

            logging.info(f"Fetched {len(recordings)} recordings from the database.")
            return recordings
        except sqlite3.Error as e:
            logging.error(f"Failed to fetch recordings: {e}")
            sys.exit(1)

    def process_recordings(self, recordings: List[CustomerRecording]):
        for recording in recordings:
            s3_key = f"{recording.master_id}/{recording.filename}"
            local_path = os.path.join(self.directory, str(recording.master_id), recording.filename)

            if not self.force and os.path.exists(local_path):
                logging.debug(f"Skipping existing file: {local_path}")
                continue

            if self.download_file(s3_key, local_path):
                self.process_audio(local_path)

    def download_file(self, s3_key: str, local_path: str) -> bool:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_path = f"s3://{self.bucket}/{s3_key}"
        command = [
            "s3cmd",
            "--config",
            self.s3cfg,
            "get",
            s3_path,
            local_path
        ]
        try:
            if self.force and os.path.exists(local_path):
                logging.debug(f"Removing existing file: {local_path}")
                os.remove(local_path)

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
        except OSError as e:
            logging.error(f"OS error occurred while handling file {local_path}: {e}")
            return False

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
        "--directory", default="customer_recordings", help="Parent directory for downloaded files (default: customer_recordings)."
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
    processor = AudioProcessor(
        bucket=args.bucket,
        s3cfg=args.s3cfg,
        db_name=args.db_name,
        directory=args.directory,
        debug=args.debug,
        force=args.force,
        limit=args.limit,
        batch_size=args.batch_size
    )
    processor.run()

if __name__ == "__main__":
    main()
