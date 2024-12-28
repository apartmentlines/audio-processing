#!/usr/bin/env python

"""
A multi-threaded transcription pipeline that simulates downloading and transcribing files.
"""

# Default configuration values
DEFAULT_TRANSCRIPTION_LIMIT = 3
DEFAULT_DOWNLOAD_QUEUE_SIZE = 10

import argparse
import json
import logging
import queue
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class FileData:
    """Dataclass representing a file to be processed."""
    id: int
    name: str
    url: str

# Walk-through
#
# populate_download_queue: Loops through file_list and pushes each file name into download_queue.
# download_files:
# Continuously pulls a filename from download_queue.
# “Downloads” it (simulated by time.sleep(1)).
# Pushes it into downloaded_queue (blocking if that queue is full).
# Calls task_done() on download_queue.
# transcribe_consumer:
# Continuously pulls a filename from downloaded_queue.
# Waits on a transcription_sema.acquire() to ensure at most 3 transcriptions run at once.
# Spawns a new thread (transcribe_file) to transcribe that file.
# Calls task_done() on downloaded_queue.
# transcribe_file:
# “Transcribes” the file (simulated by time.sleep(2)).
# Calls transcription_sema.release() in a finally block to free the transcription slot.

# Why This Works

# Unlimited Download Queue: The only limit on how many items can be queued for download is how many files you have.
# Downloaded Queue = 10: Downloading blocks if there are 10 items waiting to be transcribed. This prevents a large backlog from building up.
# Semaphore (3): Only 3 transcription threads can be active at once.
# Not Strictly FIFO: Because each transcription is its own thread, the order in which they finish can be arbitrary. You aren’t forced into a “pull from queue, transcribe in that order” model.
# Pulls Only When Slots Are Available: If all 3 transcription slots are busy, the consumer thread blocks on semaphore.acquire(), so it won’t read the next file from downloaded_queue (or at least it won’t fully process it) until a slot is freed. That, in turn, can block the downloader if the downloaded_queue gets full, giving you a natural pipeline flow.

class TranscriptionPipeline:
    """Main class for managing the transcription pipeline."""

    def __init__(
        self,
        transcription_limit: int = DEFAULT_TRANSCRIPTION_LIMIT,
        download_queue_size: int = DEFAULT_DOWNLOAD_QUEUE_SIZE,
        debug: bool = False,
    ):
        self.active_transcriptions = []
        """
        Initialize the transcription pipeline.

        :param transcription_limit: Maximum concurrent transcriptions
        :param download_queue_size: Maximum size of downloaded files queue
        :param debug: Enable debug logging
        """
        self.transcription_limit = transcription_limit
        self.download_queue_size = download_queue_size
        self.debug = debug

        self.transcription_sema = threading.Semaphore(self.transcription_limit)
        self.download_queue = queue.Queue()
        self.downloaded_queue = queue.Queue(maxsize=self.download_queue_size)
        self.shutdown_event = threading.Event()

        # Configure logging
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
        )

    def populate_download_queue(self, file_list: List[FileData]) -> None:
        """Producer that enqueues files for download.

        :param file_list: List of FileData objects to process
        """
        for file_data in file_list:
            logging.debug(f"Adding {file_data.name} to download queue.")
            self.download_queue.put(file_data)
        logging.info("Finished populating download queue.")

    def download_files(self) -> None:
        """Worker that pulls from download_queue, simulates a download,
        and pushes into downloaded_queue."""
        while not self.shutdown_event.is_set():
            try:
                file_data = self.download_queue.get()
                if file_data is None:
                    self.download_queue.task_done()
                    break
                logging.debug(f"Downloading {file_data.name} from {file_data.url}")
                time.sleep(random.uniform(1, 3))

                self.downloaded_queue.put(file_data)
                logging.info(f"Downloaded {file_data.name} from {file_data.url}")
                if self.downloaded_queue.full():
                    logging.debug("Downloaded queue is full - downloader will block")

                self.download_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error downloading {file_data.name}: {e}")
                self.download_queue.task_done()
        logging.info("Exiting download thread.")

    def transcribe_consumer(self) -> None:
        """
        A single consumer thread that pulls from downloaded_queue
        and spawns transcription threads, respecting the concurrency limit.
        """
        while not self.shutdown_event.is_set() or not self.downloaded_queue.empty() or self.active_transcriptions:
            try:
                file_data = self.downloaded_queue.get(timeout=0.1)
                if file_data is None:
                    self.downloaded_queue.task_done()
                    break
                self.transcription_sema.acquire()
                logging.debug(f"Acquired transcription slot for {file_data.name}.")

                t = threading.Thread(
                    target=self.transcribe_file,
                    args=(file_data,),
                    name=f"Transcribe-{file_data.id}",
                )
                t.start()
                self.active_transcriptions.append(t)

                self.downloaded_queue.task_done()
            except queue.Empty:
                if self.shutdown_event.is_set() and self.downloaded_queue.empty() and not self.active_transcriptions:
                    break
                continue
            except Exception as e:
                logging.error(f"Error processing {file_data.name}: {e}")
                self.downloaded_queue.task_done()

            self.active_transcriptions = [t for t in self.active_transcriptions if t.is_alive()]
        logging.info("Exiting consumer thread.")

    def transcribe_file(self, file_data: FileData) -> None:
        """Transcription function, run in its own thread.

        :param file_data: FileData object containing file information
        """
        try:
            logging.debug(f"Starting transcription for {file_data.name}")
            time.sleep(random.uniform(5, 10))
            logging.info(f"Finished transcription for {file_data.name}")
        except Exception as e:
            logging.error(f"Error transcribing {file_data.name}: {e}")
        finally:
            self.transcription_sema.release()
            logging.debug(f"Released transcription slot for {file_data.name}")

    def load_files(self, file_path: Path) -> List[FileData]:
        """Load and parse the JSON file containing files to process.

        :param file_path: Path to JSON file
        :return: List of FileData objects
        :raises: Exception if file cannot be loaded or parsed
        """
        logging.info(f"Loading files from {file_path}")
        try:
            with open(file_path) as f:
                file_data = json.load(f)
                return [FileData(id=item['id'], name=item['name'], url=item['url']) for item in file_data]
        except Exception as e:
            logging.error(f"Error loading JSON file: {e}")
            raise

    def run(self, file_path: Path) -> None:
        """Run the transcription pipeline with the given file list.

        :param file_path: Path to JSON file containing files to process
        """
        file_list = self.load_files(file_path)
        logging.info("Starting transcription pipeline...")
        try:
            download_thread = threading.Thread(
                target=self.download_files, daemon=True, name="Downloader"
            )
            consumer_thread = threading.Thread(
                target=self.transcribe_consumer, daemon=True, name="Consumer"
            )

            download_thread.start()
            consumer_thread.start()

            self.populate_download_queue(file_list)
            self.download_queue.put(None)

            self.download_queue.join()
            self.shutdown_event.set()
            self.downloaded_queue.join()

            while self.active_transcriptions:
                self.active_transcriptions = [t for t in self.active_transcriptions if t.is_alive()]
                if self.active_transcriptions:
                    time.sleep(0.1)

            self.downloaded_queue.put(None)

            download_thread.join()
            consumer_thread.join()

            logging.info("Pipeline completed successfully.")
        except KeyboardInterrupt:
            logging.info("Received interrupt signal. Shutting down gracefully...")
            self.shutdown_event.set()
            exit(0)
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            exit(1)

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run the transcription pipeline.")
    parser.add_argument(
        "--files",
        type=Path,
        required=True,
        help="Path to JSON file containing list of files to process",
    )
    parser.add_argument(
        "--transcription-limit",
        type=int,
        default=DEFAULT_TRANSCRIPTION_LIMIT,
        help="Maximum concurrent transcriptions, default %(default)s",
    )
    parser.add_argument(
        "--download-queue-size",
        type=int,
        default=DEFAULT_DOWNLOAD_QUEUE_SIZE,
        help="Maximum size of downloaded files queue, default %(default)s",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()

def main() -> None:
    """Main function to run the transcription pipeline."""
    args = parse_arguments()

    pipeline = TranscriptionPipeline(
        transcription_limit=args.transcription_limit,
        download_queue_size=args.download_queue_size,
        debug=args.debug,
    )

    pipeline.run(args.files)

if __name__ == "__main__":
    main()
