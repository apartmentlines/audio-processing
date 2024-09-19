#!/usr/bin/env python3

"""
A script to update EAF files for customer recordings and mark them as complete in the database.
"""

import argparse
import logging
import sqlite3
import sys
import time
import signal
import tarfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import subprocess
import os


@dataclass
class CustomerRecording:
    id: int
    master_id: int
    filename: str
    timestamp: int


class EAFUpdater:
    def __init__(
        self,
        db_name: str,
        eaf_directory: Path,
        debug: bool = False,
        limit: Optional[int] = None,
        batch_size: int = 100,
        archive_dir: Path = Path.home() / "Downloads",
    ):
        self.db_name = db_name
        self.eaf_directory = eaf_directory
        self.debug = debug
        self.limit = limit
        self.batch_size = batch_size
        self.conn = None
        self.setup_logging()
        self.archive_dir = archive_dir
        self.archive_path = None

    def setup_logging(self):
        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        if self.debug:
            logging.debug(f"Debug mode enabled. Arguments: {self.__dict__}")

    def get_db_connection(self):
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.conn.row_factory = sqlite3.Row
            logging.debug(f"Successfully connected to the database: {self.db_name}")
        except sqlite3.Error as e:
            logging.error(f"Failed to connect to the database: {e}")
            sys.exit(1)

    def fetch_recordings(self) -> List[CustomerRecording]:
        query = """
            SELECT id, master_id, filename, timestamp
            FROM customer_recordings
            WHERE eaf_complete = 0
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

                recordings.extend(
                    [
                        CustomerRecording(
                            id=row["id"],
                            master_id=row["master_id"],
                            filename=row["filename"],
                            timestamp=row["timestamp"],
                        )
                        for row in batch
                    ]
                )

                offset += self.batch_size

                if self.limit and len(recordings) >= self.limit:
                    recordings = recordings[: self.limit]
                    break

            logging.info(f"Fetched {len(recordings)} recordings from the database.")
            return recordings
        except sqlite3.Error as e:
            logging.error(f"Failed to fetch recordings: {e}")
            sys.exit(1)

    def get_eaf_path(self, recording: CustomerRecording) -> Path:
        return self.eaf_directory / f"{Path(recording.filename).stem}.eaf"

    def open_file(self, file_path: Path):
        if sys.platform == "darwin":
            subprocess.run(["open", str(file_path)])
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(file_path)])
        else:
            logging.error(f"Unsupported platform: {sys.platform}")
            sys.exit(1)

    def mark_complete(self, recording: CustomerRecording, original_mtime: float):
        eaf_path = self.get_eaf_path(recording)
        current_mtime = eaf_path.stat().st_mtime

        if current_mtime != original_mtime:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    "UPDATE customer_recordings SET eaf_complete = 1 WHERE id = ?",
                    (recording.id,),
                )
                self.conn.commit()
                logging.info(f"Marked recording ID {recording.id} (filename: {recording.filename}) as complete.")
                return True
            except sqlite3.Error as e:
                logging.error(f"Failed to update database for recording ID {recording.id} (filename: {recording.filename}): {e}")
                return False
        else:
            logging.warning(f"EAF file {eaf_path} has not been modified.")
            logging.warning("You must save your changes to the EAF file before marking it as complete.")
            return False

    def create_archive(self):
        current_date = time.strftime("%Y-%m-%d")
        archive_name = f"eaf_update_archive_{current_date}.tar.gz"
        self.archive_path = self.archive_dir / archive_name

        with tarfile.open(self.archive_path, "w:gz") as tar:
            tar.add(self.db_name, arcname=Path(self.db_name).name)
            tar.add(self.eaf_directory, arcname=self.eaf_directory.name)

        logging.info(f"Created archive: {self.archive_path}")

    def quit_process(self):
        logging.info("Quitting process...")
        self.create_archive()
        if self.conn:
            self.conn.close()
            logging.debug("Database connection closed.")
        logging.info(f"Archive created at: {self.archive_path}")
        sys.exit(0)

    def run(self):
        logging.info("Starting EAF update script")

        self.get_db_connection()
        recordings = self.fetch_recordings()

        for recording in recordings:
            eaf_path = self.get_eaf_path(recording)

            if not eaf_path.exists():
                logging.warning(f"EAF file not found for recording ID {recording.id} (filename: {recording.filename}): {eaf_path}")
                continue

            original_mtime = eaf_path.stat().st_mtime

            while True:
                confirm = input(f"Open EAF file for recording ID {recording.id} (filename: {recording.filename})? (y/n/q): ").lower()
                if confirm == 'y':
                    self.open_file(eaf_path)
                    break
                elif confirm == 'n':
                    logging.info(f"Skipping recording ID {recording.id} (filename: {recording.filename})")
                    self.quit_process()
                elif confirm == 'q':
                    self.quit_process()
                else:
                    print("Invalid input. Please enter 'y', 'n', or 'q'.")

            while True:
                action = input("Mark as complete (c) or quit (q)? ").lower()
                if action == 'c':
                    if self.mark_complete(recording, original_mtime):
                        break
                elif action == 'q':
                    self.quit_process()
                else:
                    print("Invalid input. Please enter 'c' or 'q'.")

        self.quit_process()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update EAF files for customer recordings and mark them as complete."
    )
    parser.add_argument(
        "--db-name",
        default="customer_recordings.db",
        help="SQLite database name (default: %(default)s).",
    )
    parser.add_argument(
        "--eaf-directory",
        type=Path,
        default=Path("eaf"),
        help="Directory containing EAF files (default: %(default)s).",
    )
    parser.add_argument(
        "--archive-dir",
        type=lambda p: Path(os.path.expanduser(p)),
        default=Path.home() / "Downloads",
        help="Directory for storing the archive (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of records to fetch in each database query (default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the total number of recordings to process.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()

def signal_handler(signum, frame):
    logging.info("Received interrupt signal.")
    updater.quit_process()

def main():
    global updater
    args = parse_arguments()
    updater = EAFUpdater(
        db_name=args.db_name,
        eaf_directory=args.eaf_directory,
        debug=args.debug,
        limit=args.limit,
        batch_size=args.batch_size,
        archive_dir=args.archive_dir,
    )

    signal.signal(signal.SIGINT, signal_handler)

    updater.run()

if __name__ == "__main__":
    main()