#!/usr/bin/env python3

"""
A script to submit diarization jobs to pyannote.ai API based on records from an SQLite database.
Also starts a local web server to serve audio files and receive webhook results.
"""

import argparse
import logging
import json
import os
import sqlite3
import sys
import time
import threading
from os import environ
from dataclasses import dataclass
from typing import List, Optional
import requests
from flask import Flask, request, send_file, jsonify, abort
from queue import Queue
from threading import Event

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
    def __init__(self,
                 api_key: Optional[str],
                 db_name: str,
                 directory: str,
                 endpoint_hostname: str,
                 endpoint_port: int = 4000,
                 debug: bool = False,
                 force: bool = False,
                 limit: Optional[int] = None,
                 batch_size: int = 100):
        self.api_key = api_key or environ.get('PYANNOTE_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either through --api-key argument or PYANNOTE_API_KEY environment variable")
        self.db_name = db_name
        self.directory = directory
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

    def make_api_request(self, url, method='GET', data=None):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, headers=headers, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return None

    def get_file_path(self, recording: CustomerRecording) -> str:
        return os.path.join(self.directory, str(recording.master_id), recording.filename)

    def get_diarization_results_path(self, recording: CustomerRecording) -> str:
        file_path = self.get_file_path(recording)
        return f"{os.path.splitext(file_path)[0]}.diarization.json"

    def validate_endpoint_hostname(self):
        import re
        # More flexible hostname validation
        hostname_regex = re.compile(
            r'^(?=.{1,255}$)([0-9A-Za-z](?:(?:[0-9A-Za-z]|-){0,61}[0-9A-Za-z])?\.)+[A-Za-z]{2,63}$'
        )
        if not hostname_regex.match(self.endpoint_hostname):
            logging.error(f"Invalid endpoint hostname: {self.endpoint_hostname}. Must be a valid hostname.")
            sys.exit(1)
        else:
            logging.debug(f"Endpoint hostname {self.endpoint_hostname} validated.")

    def setup_logging(self):
        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
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

    def setup_routes(self):
        @self.app.route('/audio/<int:recording_id>', methods=['GET'])
        def serve_audio(recording_id):
            recording = self.get_recording_by_id(recording_id)
            if not recording:
                logging.error(f"Recording not found: ID {recording_id}")
                abort(404)
            file_path = self.get_file_path(recording)
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                logging.error(f"Audio file not found: {file_path}")
                abort(404)
            logging.debug(f"Serving audio file: {file_path}")
            return send_file(file_path, mimetype='audio/wav')

        @self.app.route('/results/<int:recording_id>', methods=['POST'])
        def receive_results(recording_id):
            data = request.get_json()
            if not data:
                logging.error(f"No JSON data received for recording ID {recording_id}")
                abort(400)
            recording = self.get_recording_by_id(recording_id)
            if not recording:
                logging.error(f"Recording not found: ID {recording_id}")
                abort(404)
            diarization_results_path = self.get_diarization_results_path(recording)
            try:
                with open(diarization_results_path, 'w') as f:
                    json.dump(data, f, indent=4)
                logging.info(f"Received and saved diarization results for recording ID {recording_id} (filename: {recording.filename}) at {diarization_results_path}")
                self.job_queue.task_done()  # Mark the job as done
                return jsonify({'status': 'received'}), 200
            except Exception as e:
                logging.error(f"Failed to save diarization results for recording ID {recording_id} (filename: {recording.filename}): {e}")
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
                    id=row['id'],
                    master_id=row['master_id'],
                    filename=row['filename'],
                    timestamp=row['timestamp']
                )
            else:
                return None
        except sqlite3.Error as e:
            logging.error(f"Failed to get recording ID {recording_id}: {e}")
            return None

    def start_web_server(self):
        def run_app():
            logging.info(f"Starting web server on port {self.endpoint_port}")
            self.app.run(host='0.0.0.0', port=self.endpoint_port)
        threading.Thread(target=run_app, daemon=True).start()

    def process_recording(self, recording: CustomerRecording) -> dict:
        if self.should_skip_recording(recording):
            return {"success": True, "skipped": True}

        file_url = f'https://{self.endpoint_hostname}/audio/{recording.id}'
        webhook_url = f'https://{self.endpoint_hostname}/results/{recording.id}'

        request_data = {
            'url': file_url,
            'webhook': webhook_url
        }

        logging.info(f"Submitting diarization job for recording ID {recording.id} (filename: {recording.filename})")
        response = self.make_api_request(DIARIZE_ENDPOINT, method='POST', data=request_data)

        if response is None:
            return {"success": False, "rate_limited": False}

        if response.status_code == 200:
            logging.info(f"Successfully submitted diarization job for recording ID {recording.id} (filename: {recording.filename})")
            self.job_queue.put(recording.id)  # Add job to the queue
            return {"success": True, "rate_limited": False}
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', '60'))
            logging.warning(f"Received 429 Too Many Requests, need to retry after {retry_after} seconds")
            return {"success": False, "rate_limited": True, "retry_after": retry_after}
        else:
            logging.error(f"Failed to submit diarization job for recording ID {recording.id} (filename: {recording.filename}), status code: {response.status_code}, response: {response.text}")
            return {"success": False, "rate_limited": False}

    def should_skip_recording(self, recording: CustomerRecording) -> bool:
        file_path = self.get_file_path(recording)
        diarization_results_path = self.get_diarization_results_path(recording)

        if os.path.exists(diarization_results_path) and not self.force:
            logging.info(f"Diarization results already exist for recording ID {recording.id} (filename: {recording.filename}), skipping")
            return True

        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            logging.error(f"Audio file not found: {file_path}, skipping recording ID {recording.id} (filename: {recording.filename})")
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
                logging.info(f"Sleeping for {sleep_time:.2f} seconds due to rate limiting")
                time.sleep(sleep_time)
                rate_limit_reset_time = None

            recording = recordings[index]
            result = self.process_recording(recording)

            if result["success"] or result.get("skipped", False):
                index += 1
                retry_count = 0
            elif result["rate_limited"]:
                rate_limit_reset_time = time.time() + result["retry_after"]
                logging.warning(f"Setting rate limit reset time to {result['retry_after']} seconds from now")
            else:
                retry_count += 1
                if retry_count >= max_retries:
                    logging.error(f"Max retries reached for recording ID {recording.id} (filename: {recording.filename}), moving to next recording")
                    index += 1
                    retry_count = 0
                else:
                    logging.warning(f"Retrying recording ID {recording.id} (filename: {recording.filename}) (attempt {retry_count}/{max_retries})")

        self.all_jobs_submitted.set()  # Signal that all jobs have been submitted

    def wait_for_completion(self):
        logging.info("Waiting for all diarization jobs to complete...")
        self.all_jobs_submitted.wait()  # Wait for all jobs to be submitted
        self.job_queue.join()  # Wait for all jobs in the queue to be processed
        logging.info("All diarization jobs have completed.")

    def run(self):
        start_time = time.time()
        logging.info("Starting diarization job submission script")

        # Start the web server in another thread
        self.start_web_server()

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

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit diarization jobs to pyannote.ai API based on SQLite records."
    )
    parser.add_argument(
        "--api-key", required=False, help="pyannote.ai API key. If not provided, PYANNOTE_API_KEY environment variable will be used."
    )
    parser.add_argument(
        "--db-name", default="customer_recordings.db", help="SQLite database name (default: customer_recordings.db)."
    )
    parser.add_argument(
        "--directory", default="customer_recordings", help="Parent directory for audio files (default: customer_recordings)."
    )
    parser.add_argument(
        "--endpoint-hostname", required=True, help="Endpoint hostname for building the URLs (must be a fully qualified domain name)."
    )
    parser.add_argument(
        "--endpoint-port", type=int, default=4000, help="Port for the local web server (default: 4000)."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--force", action="store_true", help="Force processing of existing files."
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
    try:
        submitter = DiarizationJobSubmitter(
            api_key=args.api_key,
            db_name=args.db_name,
            directory=args.directory,
            endpoint_hostname=args.endpoint_hostname,
            endpoint_port=args.endpoint_port,
            debug=args.debug,
            force=args.force,
            limit=args.limit,
            batch_size=args.batch_size
        )
        submitter.run()
    except ValueError as e:
        logging.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()