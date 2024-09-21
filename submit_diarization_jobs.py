#!/usr/bin/env python3

"""
A script to submit diarization jobs to pyannote.ai API based on records from an SQLite database.
Also starts a local web server to serve audio files and receive webhook results.
"""

import argparse
import logging
import json
from pathlib import Path
import sqlite3
import sys
import time
import threading
import signal
from os import environ
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import requests
from flask import Flask, request, send_file, jsonify, abort
from queue import Queue
from threading import Event
from werkzeug.serving import make_server

# Constants
API_BASE_URL = "https://api.pyannote.ai/v1"
DIARIZE_ENDPOINT = f"{API_BASE_URL}/diarize"


@dataclass
class CustomerRecording:
    id: int
    master_id: int
    filename: str
    timestamp: int


class DiarizationJobSubmitter:
    def __init__(
        self,
        api_key: Optional[str],
        db_name: str,
        data_directory: Path,
        results_directory: Path,
        endpoint_hostname: str,
        endpoint_port: int = 4321,
        debug: bool = False,
        force: bool = False,
        limit: Optional[int] = None,
        batch_size: int = 100,
    ):
        self.api_key = api_key or environ.get("PYANNOTE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either through --api-key argument or PYANNOTE_API_KEY environment variable"
            )
        self.db_name = db_name
        self.data_directory = data_directory
        self.results_directory = results_directory
        self.endpoint_hostname = endpoint_hostname
        self.endpoint_port = endpoint_port
        self.debug = debug
        self.force = force
        self.limit = limit
        self.batch_size = batch_size
        self.conn = None
        self.setup_logging()
        self.validate_endpoint_hostname()

        # Initialize Flask app
        self.app = Flask(__name__)
        self.setup_routes()

        # Initialize job queue and event
        self.job_queue = Queue()
        self.all_jobs_submitted = Event()

        # New attributes for server status
        self.server = None
        self.server_started = threading.Event()
        self.server_status = None
        self.setup_signal_handler()

    def make_api_request(self, url, method="GET", data=None):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            if method == "GET":
                response = requests.get(url, headers=headers)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return None

    def get_file_path(self, recording: CustomerRecording) -> Path:
        return self.data_directory / recording.filename

    def get_diarization_results_path(self, recording: CustomerRecording) -> Path:
        return self.results_directory / f"{Path(recording.filename).stem}.json"

    def validate_endpoint_hostname(self):
        import re

        # More flexible hostname validation
        hostname_regex = re.compile(
            r"^(?=.{1,255}$)([0-9A-Za-z](?:(?:[0-9A-Za-z]|-){0,61}[0-9A-Za-z])?\.)+[A-Za-z]{2,63}$"
        )
        if not hostname_regex.match(self.endpoint_hostname):
            logging.error(
                f"Invalid endpoint hostname: {self.endpoint_hostname}. Must be a valid hostname."
            )
            sys.exit(1)
        else:
            logging.debug(f"Endpoint hostname {self.endpoint_hostname} validated.")

    def _validate_diarization_json(self, data: Dict[str, Any]) -> bool:
        """
        Validate the structure of the diarization JSON data.

        :param data: The loaded JSON data
        :type data: Dict[str, Any]
        :return: True if the JSON is valid, False otherwise
        :rtype: bool
        """
        if not isinstance(data, dict):
            return False
        if "jobId" not in data or "status" not in data or "output" not in data:
            return False
        if not isinstance(data["output"], dict) or "diarization" not in data["output"]:
            return False
        if not isinstance(data["output"]["diarization"], list):
            return False
        for segment in data["output"]["diarization"]:
            if not isinstance(segment, dict):
                return False
            if (
                "speaker" not in segment
                or "start" not in segment
                or "end" not in segment
            ):
                return False
            if not isinstance(segment["speaker"], str):
                return False
            if not isinstance(segment["start"], (int, float)) or not isinstance(
                segment["end"], (int, float)
            ):
                return False
            if segment["start"] >= segment["end"]:
                return False
        return True

    def setup_logging(self):
        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        if self.debug:
            logging.debug(f"Debug mode enabled. Arguments: {self.__dict__}")

    def get_db_connection(self):
        try:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            logging.debug(f"Successfully connected to the database: {self.db_name}")
            return conn
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
            with self.get_db_connection() as conn:
                while True:
                    cursor = conn.cursor()
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

    def setup_routes(self):
        @self.app.route("/audio/<int:recording_id>", methods=["GET"])
        def serve_audio(recording_id):
            recording = self.get_recording_by_id(recording_id)
            if not recording:
                logging.error(f"Recording not found: ID {recording_id}")
                abort(404)
            file_path = self.get_file_path(recording)
            if not file_path.exists() or not file_path.is_file():
                logging.error(f"Audio file not found: {file_path}")
                abort(404)
            logging.debug(f"Serving audio file: {file_path}")
            return send_file(file_path.resolve(), mimetype="audio/wav")

        @self.app.route("/results/<int:recording_id>", methods=["POST"])
        def receive_results(recording_id):
            recording = self.get_recording_by_id(recording_id)
            if not recording:
                logging.error(f"Recording not found: ID {recording_id}")
                abort(404)
            data = request.get_json()
            if not self._validate_diarization_json(data):
                logging.error(
                    f"Invalid JSON data received for recording ID {recording_id} (filename: {recording.filename})"
                )
                abort(400)
            diarization_results_path = self.get_diarization_results_path(recording)
            try:
                diarization_results_path.parent.mkdir(parents=True, exist_ok=True)
                with diarization_results_path.open("w") as f:
                    json.dump(data, f, indent=4)
                logging.info(
                    f"Received and saved diarization results for recording ID {recording_id} (filename: {recording.filename}) at {diarization_results_path}"
                )
                self.job_queue.task_done()  # Mark the job as done
                return jsonify({"status": "received"}), 200
            except Exception as e:
                logging.error(
                    f"Failed to save diarization results for recording ID {recording_id} (filename: {recording.filename}): {e}"
                )
                abort(500)

    def get_recording_by_id(self, recording_id: int) -> Optional[CustomerRecording]:
        query = "SELECT id, master_id, filename, timestamp FROM customer_recordings WHERE id = ?"
        try:
            with self.conn:  # This ensures proper transaction handling
                cursor = self.conn.cursor()
                cursor.execute(query, (recording_id,))
                row = cursor.fetchone()
            if row:
                return CustomerRecording(
                    id=row["id"],
                    master_id=row["master_id"],
                    filename=row["filename"],
                    timestamp=row["timestamp"],
                )
            else:
                return None
        except sqlite3.Error as e:
            logging.error(f"Failed to get recording ID {recording_id}: {e}")
            return None

    def start_web_server(self):
        def run_app():
            try:
                logging.info(f"Starting web server on port {self.endpoint_port}")
                self.server = make_server('0.0.0.0', self.endpoint_port, self.app)
                self.server_status = True
                self.server_started.set()
                self.server.serve_forever()
            except Exception as e:
                logging.error(f"Failed to start web server: {e}")
                self.server_status = False
                self.server_started.set()

        threading.Thread(target=run_app, daemon=True).start()

    def wait_for_server_start(self, timeout=10):
        if self.server_started.wait(timeout):
            if self.server_status:
                logging.info("Web server started successfully")
                return True
            else:
                logging.error("Web server failed to start")
                return False
        else:
            logging.error("Timeout waiting for web server to start")
            return False

    def stop_web_server(self):
        if self.server:
            logging.info("Stopping web server...")
            self.server.shutdown()
            self.server.server_close()
            logging.info("Web server stopped.")

    def setup_signal_handler(self):
        def signal_handler(signum, frame):
            logging.info("Received interrupt signal. Stopping server and exiting...")
            self.stop_web_server()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def process_recording(self, recording: CustomerRecording) -> dict:
        if self.should_skip_recording(recording):
            return {"success": True, "skipped": True}

        file_url = f"https://{self.endpoint_hostname}/audio/{recording.id}"
        webhook_url = f"https://{self.endpoint_hostname}/results/{recording.id}"

        request_data = {"url": file_url, "webhook": webhook_url}

        logging.info(
            f"Submitting diarization job for recording ID {recording.id} (filename: {recording.filename})"
        )
        response = self.make_api_request(
            DIARIZE_ENDPOINT, method="POST", data=request_data
        )

        if response is None:
            return {"success": False, "rate_limited": False}

        if response.status_code == 200:
            logging.info(
                f"Successfully submitted diarization job for recording ID {recording.id} (filename: {recording.filename})"
            )
            self.job_queue.put(recording.id)  # Add job to the queue
            return {"success": True, "rate_limited": False}
        elif response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "60"))
            logging.warning(
                f"Received 429 Too Many Requests, need to retry after {retry_after} seconds"
            )
            return {"success": False, "rate_limited": True, "retry_after": retry_after}
        else:
            logging.error(
                f"Failed to submit diarization job for recording ID {recording.id} (filename: {recording.filename}), status code: {response.status_code}, response: {response.text}"
            )
            return {"success": False, "rate_limited": False}

    def should_skip_recording(self, recording: CustomerRecording) -> bool:
        file_path = self.get_file_path(recording)
        diarization_results_path = self.get_diarization_results_path(recording)

        if diarization_results_path.exists() and not self.force:
            logging.info(
                f"Diarization results already exist for recording ID {recording.id} (filename: {recording.filename}), skipping"
            )
            return True

        if not file_path.exists() or not file_path.is_file():
            logging.error(
                f"Audio file not found: {file_path}, skipping recording ID {recording.id} (filename: {recording.filename})"
            )
            return True

        return False

    def process_recordings(self, recordings: List[CustomerRecording]):
        rate_limit_reset_time = None
        index = 0
        max_retries = 5
        retry_count = 0

        while index < len(recordings):
            if rate_limit_reset_time and time.time() < rate_limit_reset_time:
                sleep_time = rate_limit_reset_time - time.time()
                logging.info(
                    f"Sleeping for {sleep_time:.2f} seconds due to rate limiting"
                )
                time.sleep(sleep_time)
                rate_limit_reset_time = None

            recording = recordings[index]
            result = self.process_recording(recording)

            if result["success"] or result.get("skipped", False):
                index += 1
                retry_count = 0
            elif result["rate_limited"]:
                rate_limit_reset_time = time.time() + result["retry_after"]
                logging.warning(
                    f"Setting rate limit reset time to {result['retry_after']} seconds from now"
                )
            else:
                retry_count += 1
                if retry_count >= max_retries:
                    logging.error(
                        f"Max retries reached for recording ID {recording.id} (filename: {recording.filename}), moving to next recording"
                    )
                    index += 1
                    retry_count = 0
                else:
                    logging.warning(
                        f"Retrying recording ID {recording.id} (filename: {recording.filename}) (attempt {retry_count}/{max_retries})"
                    )

        self.all_jobs_submitted.set()  # Signal that all jobs have been submitted

    def wait_for_completion(self):
        logging.info("Waiting for all diarization jobs to complete...")
        self.all_jobs_submitted.wait()  # Wait for all jobs to be submitted
        self.job_queue.join()  # Wait for all jobs in the queue to be processed
        logging.info("All diarization jobs have completed.")

    def run(self):
        start_time = time.time()
        logging.info("Starting diarization job submission script")

        try:
            # Start the web server in another thread
            self.start_web_server()

            # Wait for the server to start and check its status
            if not self.wait_for_server_start():
                logging.error("Exiting due to web server startup failure")
                sys.exit(1)  # Exit with error code

            self.conn = self.get_db_connection()
            recordings = self.fetch_recordings()
            self.process_recordings(recordings)

            self.wait_for_completion()  # Wait for all jobs to complete

            if self.conn:
                self.conn.close()
                logging.debug("Database connection closed.")

            end_time = time.time()
            logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")
            logging.info("Diarization job submission script completed")
        finally:
            self.stop_web_server()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit diarization jobs to pyannote.ai API based on SQLite records."
    )
    parser.add_argument(
        "--db-name",
        default="customer_recordings.db",
        help="SQLite database name (default: %(default)s).",
    )
    parser.add_argument(
        "--api-key",
        required=False,
        help="pyannote.ai API key. If not provided, PYANNOTE_API_KEY environment variable will be used.",
    )
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path("audio"),
        help="Parent directory for audio files (default: %(default)s).",
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=Path("diarization-results"),
        help="Directory for storing diarization results (default: %(default)s).",
    )
    parser.add_argument(
        "--endpoint-hostname",
        required=True,
        help="Endpoint hostname for building the URLs (must be a fully qualified domain name).",
    )
    parser.add_argument(
        "--endpoint-port",
        type=int,
        default=4321,
        help="Port for the local web server (default: %(default)s).",
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
    parser.add_argument(
        "--force", action="store_true", help="Force processing of existing files."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    try:
        submitter = DiarizationJobSubmitter(
            api_key=args.api_key,
            db_name=args.db_name,
            data_directory=args.data_directory,
            results_directory=args.results_directory,
            endpoint_hostname=args.endpoint_hostname,
            endpoint_port=args.endpoint_port,
            debug=args.debug,
            force=args.force,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        submitter.run()
    except ValueError as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
