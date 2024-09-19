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
from dataclasses import dataclass
from typing import List, Optional
import requests
from flask import Flask, request, send_file, jsonify, abort

@dataclass
class CustomerRecording:
    id: int
    master_id: int
    filename: str
    timestamp: int

class DiarizationJobSubmitter:
    def __init__(self,
                 api_key: str,
                 db_name: str,
                 directory: str,
                 hostname: str,
                 endpoint_port: int = 4000,
                 debug: bool = False,
                 force: bool = False,
                 limit: Optional[int] = None,
                 batch_size: int = 100):
        self.api_key = api_key
        self.db_name = db_name
        self.directory = directory
        self.hostname = hostname
        self.endpoint_port = endpoint_port
        self.debug = debug
        self.force = force
        self.limit = limit
        self.batch_size = batch_size
        self.conn = None
        self.setup_logging()
        self.validate_hostname()

        # Initialize Flask app
        self.app = Flask(__name__)
        self.setup_routes()

    def validate_hostname(self):
        import re
        # Simple FQDN validation
        fqdn_regex = re.compile(
            r'^(?=.{1,255}$)([0-9A-Za-z]{1,63}\.)+([A-Za-z]{2,63})$'
        )
        if not fqdn_regex.match(self.hostname):
            logging.error(f"Invalid hostname: {self.hostname}. Must be a fully qualified domain name.")
            sys.exit(1)
        else:
            logging.debug(f"Hostname {self.hostname} validated as FQDN.")

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
                logging.error(f"Recording not found: {recording_id}")
                abort(404)
            file_path = os.path.join(self.directory, str(recording.master_id), recording.filename)
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                logging.error(f"Audio file not found: {file_path}")
                abort(404)
            logging.debug(f"Serving audio file: {file_path}")
            return send_file(file_path, mimetype='audio/wav')

        @self.app.route('/results/<int:recording_id>', methods=['POST'])
        def receive_results(recording_id):
            data = request.get_json()
            if not data:
                logging.error(f"No JSON data received for recording {recording_id}")
                abort(400)
            recording = self.get_recording_by_id(recording_id)
            if not recording:
                logging.error(f"Recording not found: {recording_id}")
                abort(404)
            file_path = os.path.join(self.directory, str(recording.master_id), recording.filename)
            output_path = f"{file_path}.diarization.json"
            try:
                with open(output_path, 'w') as f:
                    json.dump(data, f)
                logging.info(f"Received and saved diarization results for recording {recording_id} at {output_path}")
                return jsonify({'status': 'received'}), 200
            except Exception as e:
                logging.error(f"Failed to save diarization results for recording {recording_id}: {e}")
                abort(500)

    def get_recording_by_id(self, recording_id: int) -> Optional[CustomerRecording]:
        query = "SELECT id, master_id, filename, timestamp FROM customer_recordings WHERE id = ?"
        try:
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
            logging.error(f"Failed to get recording {recording_id}: {e}")
            return None

    def start_web_server(self):
        def run_app():
            logging.info(f"Starting web server on port {self.endpoint_port}")
            self.app.run(host='0.0.0.0', port=self.endpoint_port)
        threading.Thread(target=run_app, daemon=True).start()

    def process_recordings(self, recordings: List[CustomerRecording]):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        api_url = "https://api.pyannote.ai/v1/diarize"
        rate_limit_reset_time = None

        for recording in recordings:
            file_path = os.path.join(self.directory, str(recording.master_id), recording.filename)
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                logging.error(f"Audio file not found: {file_path}, skipping recording {recording.id}")
                continue

            file_url = f'https://{self.hostname}/audio/{recording.id}'
            webhook_url = f'https://{self.hostname}/results/{recording.id}'

            request_data = {
                'url': file_url,
                'webhook': webhook_url
            }

            # Rate limiting handling
            if rate_limit_reset_time and time.time() < rate_limit_reset_time:
                sleep_time = rate_limit_reset_time - time.time()
                logging.info(f"Sleeping for {sleep_time:.2f} seconds due to rate limiting")
                time.sleep(sleep_time)
                rate_limit_reset_time = None

            try:
                logging.info(f"Submitting diarization job for recording {recording.id}")
                response = requests.post(api_url, headers=headers, json=request_data)
                if response.status_code == 200:
                    logging.info(f"Successfully submitted diarization job for recording {recording.id}")
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', '60'))
                    rate_limit_reset_time = time.time() + retry_after
                    logging.warning(f"Received 429 Too Many Requests, retrying after {retry_after} seconds")
                    time.sleep(retry_after)
                    continue  # Retry the current recording
                else:
                    logging.error(f"Failed to submit diarization job for recording {recording.id}, status code: {response.status_code}, response: {response.text}")
            except requests.exceptions.RequestException as e:
                logging.error(f"Request exception for recording {recording.id}: {e}")

    def run(self):
        start_time = time.time()
        logging.info("Starting diarization job submission script")

        # Start the web server in another thread
        self.start_web_server()

        self.get_db_connection()
        recordings = self.fetch_recordings()
        self.process_recordings(recordings)

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
        "--api-key", required=True, help="pyannote.ai API key."
    )
    parser.add_argument(
        "--db-name", default="customer_recordings.db", help="SQLite database name (default: customer_recordings.db)."
    )
    parser.add_argument(
        "--directory", default="customer_recordings", help="Parent directory for audio files (default: customer_recordings)."
    )
    parser.add_argument(
        "--hostname", required=True, help="Hostname for building the URLs (must be a fully qualified domain name)."
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
    submitter = DiarizationJobSubmitter(
        api_key=args.api_key,
        db_name=args.db_name,
        directory=args.directory,
        hostname=args.hostname,
        endpoint_port=args.endpoint_port,
        debug=args.debug,
        force=args.force,
        limit=args.limit,
        batch_size=args.batch_size
    )
    submitter.run()

if __name__ == "__main__":
    main()
