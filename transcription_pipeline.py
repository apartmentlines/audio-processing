#!/usr/bin/env python

"""
A multi-threaded processing pipeline that downloads and processes files.
"""

import argparse
import importlib
import json
import logging
import queue
import random
import tempfile
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Type, Union
from tenacity import retry, stop_after_attempt, wait_exponential


# Default configuration values
DEFAULT_PROCESSING_LIMIT = 3
DEFAULT_DOWNLOAD_QUEUE_SIZE = 10
DEFAULT_MAX_RETRIES = 3
DOWNLOAD_TIMEOUT = 30
SIMULATE_MIN_PROCESS_TIME = 5
SIMULATE_MAX_PROCESS_TIME = 10


@dataclass
class FileData:
    """Dataclass representing a file to be processed."""
    url: str
    id: Optional[Union[str, int]] = None
    name: Optional[str] = None
    local_path: Optional[Path] = None


class BaseProcessor:
    """
    Base class for processing files.
    Users should inherit from this class and implement the `process` method.
    """

    def process(self, file_data: FileData) -> Any:
        """
        Process the given file_data.

        :param file_data: FileData object containing file information
        :return: Result of the processing
        """
        raise NotImplementedError("Subclasses must implement this method.")


class BasePostProcessor:
    """
    Base class for post-processing results.
    Users should inherit from this class and implement the `post_process` method.
    """

    def post_process(self, result: Any) -> None:
        """
        Post-process the given result.

        :param result: Result returned from the processor
        """
        raise NotImplementedError("Subclasses must implement this method.")


class DummyProcessor(BaseProcessor):
    """Dummy processor that simulates file processing."""

    def process(self, file_data: FileData) -> str:
        time.sleep(random.uniform(SIMULATE_MIN_PROCESS_TIME, SIMULATE_MAX_PROCESS_TIME))
        return f"Dummy processed file {file_data.name}"


class DummyPostProcessor(BasePostProcessor):
    """Dummy post-processor that simulates post-processing."""

    def post_process(self, result: Any) -> None:
        logging.debug(f"Dummy post-processing result: {result}")


class ProcessingPipeline:
    """Main class for managing the processing pipeline."""

    def __init__(
        self,
        processor_class: Type[BaseProcessor] = DummyProcessor,
        post_processor_class: Type[BasePostProcessor] = DummyPostProcessor,
        processing_limit: int = DEFAULT_PROCESSING_LIMIT,
        download_queue_size: int = DEFAULT_DOWNLOAD_QUEUE_SIZE,
        download_cache: Path = Path(tempfile.gettempdir()) / "processing-pipeline-download-cache",
        simulate_downloads: bool = False,
        debug: bool = False,
    ):
        """
        Initialize the processing pipeline.

        :param processor_class: Class to use for processing files
        :param post_processor_class: Class to use for post-processing results
        :param processing_limit: Maximum concurrent processing threads
        :param download_queue_size: Maximum size of downloaded files queue
        :param simulate_downloads: Whether to simulate downloads
        :param debug: Enable debug logging
        """
        self.processor_class = processor_class
        self.post_processor_class = post_processor_class
        self.processing_limit = processing_limit
        self.download_queue_size = download_queue_size
        self.simulate_downloads = simulate_downloads
        self.debug = debug

        self.download_cache = download_cache
        self.download_cache.mkdir(parents=True, exist_ok=True)

        self.executor: Optional[ThreadPoolExecutor] = None
        self.download_queue: queue.Queue[Optional[FileData]] = queue.Queue()
        self.downloaded_queue: queue.Queue[Optional[FileData]] = queue.Queue(maxsize=self.download_queue_size)
        self.shutdown_event: threading.Event = threading.Event()

        self.post_processing_queue: queue.Queue[Any] = queue.Queue()

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
        try:
            while not self.shutdown_event.is_set():
                try:
                    file_data = self.download_queue.get()
                    if file_data is None:
                        break
                    self.handle_download(file_data)
                    self.downloaded_queue.put(file_data)
                    if self.downloaded_queue.full():
                        logging.debug("Downloaded queue is full - downloader will block")
                except Exception as e:
                    logging.error(f"Failed to download {file_data.name}: {e}")
        finally:
            self.downloaded_queue.put(None)
            logging.info("Exiting download thread.")

    def handle_download(self, file_data: FileData) -> None:
        """Handle the download of a file, either simulated or actual.

        :param file_data: FileData object containing file information
        """
        if self.simulate_downloads:
            self.simulate_download(file_data)
        else:
            self.perform_actual_download(file_data)

    def simulate_download(self, file_data: FileData) -> None:
        """Simulate a download by sleeping and setting a local path.

        :param file_data: FileData object containing file information
        """
        logging.debug(f"Simulating download of {file_data.name} from {file_data.url}")
        time.sleep(random.uniform(1, 3))
        file_data.local_path = Path(file_data.name)
        logging.info(f"Simulated download of {file_data.name}")

    def perform_actual_download(self, file_data: FileData) -> None:
        """Perform an actual download of the file.

        :param file_data: FileData object containing file information
        """
        local_path = self.create_cached_download_filepath(file_data)
        file_data.local_path = local_path
        self.download_file(file_data)

    @retry(
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def download_file(self, file_data: FileData) -> None:
        try:
            response = requests.get(file_data.url, timeout=DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            with open(file_data.local_path, "wb") as f:
                f.write(response.content)
            logging.info(f"Downloaded {file_data.name} from {file_data.url} to {file_data.local_path}")
        except Exception as e:
            logging.warning(f"Error downloading {file_data.name}: {e}. Retrying...")
            raise

    def create_cached_download_filepath(self, file_data: FileData) -> Path:
        """Create a temporary file for the download and return its path.

        :param file_data: FileData object containing file information
        :return: Path to the temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(
            dir=self.download_cache,
            delete=False,
            prefix=f"{file_data.id}_",
            suffix=".tmp"
        )
        temp_file.close()
        local_path = Path(temp_file.name)
        return local_path

    def delete_cached_download_file(self, file_data: FileData) -> None:
        """Delete the temporary file for the download.

        :param file_data: FileData object containing file information
        """
        if not self.simulate_downloads and file_data.local_path and file_data.local_path.exists():
            try:
                file_data.local_path.unlink()
                logging.debug(f"Deleted cached file: {file_data.local_path}")
            except Exception as e:
                logging.warning(f"Failed to delete cached file {file_data.local_path}: {e}")

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
            processor = self.processor_class()
            logging.debug(f"Starting processing for {file_data.name}")
            processing_result = processor.process(file_data)
            logging.info(f"Finished processing for {file_data.name}")
            self.post_processing_queue.put(processing_result)
            logging.debug(f"Enqueued processing result for {file_data.name} to post-processing queue")
            self.delete_cached_download_file(file_data)
        except Exception as e:
            logging.error(f"Error processing {file_data.name}: {e}")

    def post_processor(self) -> None:
        """Process the processing results from the post-processing queue."""
        while not self.shutdown_event.is_set() or not self.post_processing_queue.empty():
            try:
                result = self.post_processing_queue.get(timeout=1)
                post_processor = self.post_processor_class()
                logging.debug(f"Starting post-processing for result: {result}")
                post_processor.post_process(result)
                logging.info(f"Finished post-processing for result: {result}")
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in post-processing: {e}")
        logging.info("Exiting post-processing thread.")

    def create_file_data(self, file_dict: dict) -> FileData:
        """Create a FileData object from a dictionary, using defaults for optional missing fields.

        :param file_dict: Dictionary containing file information
        :return: FileData object with all fields populated
        :raises KeyError: If URL is not provided in the input dictionary
        """
        try:
            url = file_dict["url"]
        except KeyError:
            raise KeyError(f"Required 'url' not provided for file: {file_dict}")
        basename = Path(url).stem
        file_id = file_dict.get("id", basename)
        name = file_dict.get("name", basename)
        return FileData(
            id=file_id,
            name=name,
            url=url
        )


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
                return [self.create_file_data(item) for item in file_data]
        except KeyError as e:
            logging.error(f"Invalid file entry: {str(e)}")
            exit(1)
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
        "--processor",
        type=str,
        help="Custom processing class in 'package:ClassName' format",
    )
    parser.add_argument(
        "--post-processor",
        type=str,
        help="Custom post-processing class in 'package:ClassName' format",
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
        "--download-cache",
        type=Path,
        default=Path(tempfile.gettempdir()) / "processing-pipeline-download-cache",
        help="Directory to cache downloaded files, default %(default)s",
    )
    parser.add_argument(
        "--simulate-downloads",
        action="store_true",
        help="Simulate downloads instead of performing actual downloads",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def validate_processor_class(class_path: str, base_class: Type) -> Type:
    """
    Validate and load a processor class from a module path.

    :param class_path: String in format 'module:ClassName'
    :param base_class: Base class that the loaded class must inherit from
    :return: The validated class
    :raises RuntimeError: If the class cannot be imported or doesn't inherit from base_class
    """
    try:
        module_name, class_name = class_path.split(':')
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        if not issubclass(cls, base_class):
            raise RuntimeError(f"The class '{class_path}' must inherit from {base_class.__name__}.")
        return cls
    except (ValueError, ModuleNotFoundError, AttributeError) as e:
        raise RuntimeError(f"Cannot load class '{class_path}': {e}") from e

def main() -> int:
    """Main function to run the processing pipeline.

    :return: Exit code (0 for success, non-zero for failure)
    """
    args = parse_arguments()

    processor_class = DummyProcessor
    post_processor_class = DummyPostProcessor

    if args.processor:
        try:
            processor_class = validate_processor_class(args.processor, BaseProcessor)
        except RuntimeError as e:
            logging.error(e)
            return 1

    if args.post_processor:
        try:
            post_processor_class = validate_processor_class(args.post_processor, BasePostProcessor)
        except RuntimeError as e:
            logging.error(e)
            return 1

    # Create pipeline with validated classes
    pipeline = ProcessingPipeline(
        processor_class=processor_class,
        post_processor_class=post_processor_class,
        processing_limit=args.processing_limit,
        download_queue_size=args.download_queue_size,
        download_cache=args.download_cache,
        simulate_downloads=args.simulate_downloads,
        debug=args.debug,
    )

    return pipeline.run(args.files)


if __name__ == "__main__":
     exit(main())
