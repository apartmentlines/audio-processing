# Audio Processing and Analysis Pipeline

## Overview

This project provides a comprehensive suite of tools for processing, transcribing, and analyzing audio recordings, with a focus on customer conversations or similar applications. It combines various technologies and APIs to offer a robust pipeline for handling large volumes of audio data.

## Features

- Audio file download and preprocessing
- Speech transcription and diarization
- Speaker embedding extraction
- Integration with cloud storage (S3) and APIs (pyannote.ai)
- Batch processing capabilities
- Local web server for job management

## Requirements

- Python 3.7+
- Sox (for audio processing)
- s3cmd (for S3 interactions)
- Various Python libraries (see requirements.txt files)

## Setup

1. Clone this repository
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Ensure Sox and s3cmd are installed on your system
4. Configure your S3 credentials in the .s3cfg file
5. Set up your database (SQLite, by default named `customer_recordings.db`)

   Example INSERT statements for the `customer_recordings` table:
   ```sql
   INSERT INTO customer_recordings (master_id, filename, timestamp)
   VALUES (1001, 'customer_call_1001.wav', 1632150000);

   INSERT INTO customer_recordings (master_id, filename, timestamp)
   VALUES (1002, 'customer_call_1002.wav', 1632236400);
   ```

## Usage

Each script in this project can be run independently or as part of a larger pipeline. Refer to individual script documentation for specific usage instructions.

### Example pipeline

```sh
    download_audio_files.py --bucket some-s3-bucket --s3cfg /home/someuser/.s3cfg.someconfig  --directory audio --no-subdirs --batch-size 1000 --limit 2 --debug
    submit_diarization_jobs.py --endpoint-hostname 2df7-45-46-89-55.ngrok-free.app --debug
    # Use https://github.com/apartmentlines/diarization-to-eaf to generate .eaf files suitable
    # for opening in ELAN.
    #   diarization-to-eaf --input-dir diarization-results --output-dir eaf --media-dir audio --debug
    # Use update_eafs.py to iterate through and hand tweak .eaf files
    eaf_to_rttm.py --debug
    generate_uem_for_audio_data.py --debug
```

## Contributing

Contributions to improve or extend the functionality of this project are welcome. Please submit pull requests or open issues for any bugs or feature requests.

## License

[Specify your license here]
