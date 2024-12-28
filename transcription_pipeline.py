#!/usr/bin/env python

"""
A multi-threaded processing pipeline that downloads and processes files.
"""

import argparse
import json
import logging
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List


# Default configuration values
DEFAULT_PROCESSING_LIMIT = 3
DEFAULT_DOWNLOAD_QUEUE_SIZE = 10


@dataclass
class FileData:
    """Dataclass representing a file to be processed."""
    id: int
    name: str
    url: str


class ProcessingPipeline:
    """Main class for managing the processing pipeline."""

    def __init__(
        self,
        processing_limit: int = DEFAULT_PROCESSING_LIMIT,
        download_queue_size: int = DEFAULT_DOWNLOAD_QUEUE_SIZE,
        debug: bool = False,
    ):
        """
        Initialize the processing pipeline.

        :param processing_limit: Maximum concurrent processing threads
        :param download_queue_size: Maximum size of downloaded files queue
        :param debug: Enable debug logging
        """
        self.processing_limit = processing_limit
        self.download_queue_size = download_queue_size
        self.debug = debug

        self.executor = None
        self.download_queue = queue.Queue()
        self.downloaded_queue = queue.Queue(maxsize=self.download_queue_size)
        self.shutdown_event = threading.Event()

        self.post_processing_queue = queue.Queue()

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
        try:
            while not self.shutdown_event.is_set():
                try:
                    file_data = self.download_queue.get()
                    if file_data is None:
                        break
                    logging.debug(f"Downloading {file_data.name} from {file_data.url}")
                    time.sleep(random.uniform(1, 3))  # Simulate download
                    self.downloaded_queue.put(file_data)
                    logging.info(f"Downloaded {file_data.name} from {file_data.url}")
                    if self.downloaded_queue.full():
                        logging.debug("Downloaded queue is full - downloader will block")
                except Exception as e:
                    logging.error(f"Error downloading {file_data.name}: {e}")
        finally:
            self.downloaded_queue.put(None)
            logging.info("Exiting download thread.")

    def processing_consumer(self) -> None:
        """Consumer that pulls from downloaded_queue and submits processing tasks."""
        while not self.shutdown_event.is_set():
            try:
                file_data = self.downloaded_queue.get()
                if file_data is None:
                    logging.info("All files submitted for processing.")
                    break
                self.executor.submit(self.process_file, file_data)
                logging.debug(f"Submitted processing task for {file_data.name}")
            except Exception as e:
                logging.error(f"Error processing downloaded file: {e}")

        logging.info("Exiting processing thread.")

    def process_file(self, file_data: FileData) -> None:
        """Processing function, executed by the ThreadPoolExecutor.

        :param file_data: FileData object containing file information
        """
        try:
            if self.shutdown_event.is_set():
                logging.debug(f"Shutdown event set. Skipping processing for {file_data.name}")
                return
            logging.debug(f"Starting processing for {file_data.name}")
            time.sleep(random.uniform(5, 10))  # Simulate processing
            logging.info(f"Finished processing for {file_data.name}")

            # Simulate processing result
            processing_result = f"Processed file {file_data.name}"

            # Enqueue the result for post-processing
            self.post_processing_queue.put(processing_result)
            logging.debug(f"Enqueued processing result for {file_data.name} to post-processing queue")
        except Exception as e:
            logging.error(f"Error processing {file_data.name}: {e}")

    def post_processor(self) -> None:
        """Process the processing results from the post-processing queue."""
        while not self.shutdown_event.is_set() or not self.post_processing_queue.empty():
            try:
                result = self.post_processing_queue.get(timeout=1)
                logging.debug(f"Starting post-processing for result: {result}")
                time.sleep(random.uniform(2, 4))  # Simulate post-processing
                logging.info(f"Finished post-processing for result: {result}")
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in post-processing: {e}")
        logging.info("Exiting post-processing thread.")

    def load_files(self, file_path: Path) -> List[FileData]:
        """Load and parse the JSON file containing files to process.

        :param file_path: Path to JSON file
        :return: List of FileData objects
        :raises Exception: If file cannot be loaded or parsed
        """
        logging.info(f"Loading files from {file_path}")
        try:
            with open(file_path) as f:
                file_data = json.load(f)
                return [
                    FileData(id=item["id"], name=item["name"], url=item["url"])
                    for item in file_data
                ]
        except Exception as e:
            logging.error(f"Error loading JSON file: {e}")
            raise

    def run(self, file_path: Path) -> int:
        """Run the processing pipeline with the given file list.

        :param file_path: Path to JSON file containing files to process
        :return: Exit code (0 for success, non-zero for failure)
        """
        logging.info("Starting processing pipeline...")

        file_list = self.load_files(file_path)

        post_processor_thread = threading.Thread(
            target=self.post_processor,
            name="PostProcessor",
            daemon=True,
        )
        download_thread = threading.Thread(
            target=self.download_files,
            daemon=True,
            name="Downloader",
        )
        processing_thread = threading.Thread(
            target=self.processing_consumer,
            daemon=True,
            name="Processor",
        )

        with ThreadPoolExecutor(max_workers=self.processing_limit) as executor:
            self.executor = executor

            post_processor_thread.start()
            download_thread.start()
            processing_thread.start()

            try:

                self.populate_download_queue(file_list)
                self.download_queue.put(None)  # Signal that no more files will be added

                download_thread.join()
                processing_thread.join()
            except KeyboardInterrupt:
                logging.info("Received interrupt signal. Shutting down gracefully...")
                self.shutdown_event.set()

                download_thread.join()
                processing_thread.join()
                post_processor_thread.join()

                return 130
            except Exception as e:
                logging.error(f"Pipeline failed: {e}")
                return 1

        # Signal the post_processor to shutdown
        self.shutdown_event.set()
        logging.debug("Signaled shutdown event.")

        post_processor_thread.join()

        logging.info("Pipeline completed successfully.")
        return 0


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run the processing pipeline.")
    parser.add_argument(
        "--files",
        type=Path,
        required=True,
        help="Path to JSON file containing list of files to process",
    )
    parser.add_argument(
        "--processing-limit",
        type=int,
        default=DEFAULT_PROCESSING_LIMIT,
        help="Maximum concurrent processing threads, default %(default)s",
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


def main() -> int:
    """Main function to run the processing pipeline.

    :return: Exit code (0 for success, non-zero for failure)
    """
    args = parse_arguments()

    pipeline = ProcessingPipeline(
        processing_limit=args.processing_limit,
        download_queue_size=args.download_queue_size,
        debug=args.debug,
    )

    return pipeline.run(args.files)


if __name__ == "__main__":
     exit(main())
