#!/usr/bin/env python

import queue
import random
import threading
import time

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

# Shared concurrency limit for transcriptions.
TRANSCRIPTION_LIMIT = 3
transcription_sema = threading.Semaphore(TRANSCRIPTION_LIMIT)

# Queues
download_queue = queue.Queue()               # Unbounded queue for files to be downloaded
downloaded_queue = queue.Queue(maxsize=10)   # Holds up to 10 downloaded files

def populate_download_queue(file_list):
    """Producer that enqueues files for download."""
    for f in file_list:
        print(f"[POPULATE] Adding {f} to download queue.")
        download_queue.put(f)
    print("[POPULATE] Finished populating download queue.")

def download_files():
    """Worker that pulls from 'download_queue', simulates a download,
       and pushes into 'downloaded_queue' (maxsize=10)."""
    while True:
        file = download_queue.get()
        if file is None:
            break
        print(f"[DOWNLOAD] Downloading {file}")
        time.sleep(random.uniform(1, 3))

        downloaded_queue.put(file)  # Blocks if downloaded_queue is full
        print(f"[DOWNLOAD] Downloaded {file} -> downloaded_queue")

        download_queue.task_done()
    print("[DOWNLOAD] Exiting download thread.")

def transcribe_consumer():
    """
    A single 'consumer' thread that pulls from 'downloaded_queue'
    and spawns a transcription thread for each file, respecting the concurrency limit.
    """
    while True:
        file = downloaded_queue.get()
        if file is None:
            break

        # Acquire a slot for transcription.
        transcription_sema.acquire()
        print(f"[CONSUMER] Acquired a transcription slot for {file}.")

        # Start a new thread to handle transcription.
        t = threading.Thread(target=transcribe_file, args=(file,))
        t.start()

        downloaded_queue.task_done()

    print("[CONSUMER] Exiting consumer thread.")

def transcribe_file(file):
    """Actual transcription function, run in its own thread.
       Release the semaphore when finished."""
    try:
        print(f"[TRANSCRIBE] Starting transcription for {file}")
        # Simulate a GPU-bound transcription
        time.sleep(random.uniform(3, 5))
        print(f"[TRANSCRIBE] Finished transcription for {file}")
    finally:
        # Important: release the slot so another transcription can start
        transcription_sema.release()
        print(f"[TRANSCRIBE] Released transcription slot for {file}")

def main():
    file_list = [f"file_{i}.mp3" for i in range(1, 21)]

    # Create threads for each stage
    download_thread = threading.Thread(target=download_files, daemon=True)
    consumer_thread = threading.Thread(target=transcribe_consumer, daemon=True)

    # Start the threads
    download_thread.start()
    consumer_thread.start()

    # Populate the download queue
    populate_download_queue(file_list)

    # Wait for all downloads to complete
    download_queue.join()

    # Signal the download worker to exit
    download_queue.put(None)
    download_thread.join()

    # Wait for all downloaded items to be processed
    downloaded_queue.join()

    # Signal the consumer to exit
    downloaded_queue.put(None)
    consumer_thread.join()

    print("[MAIN] All done.")

if __name__ == "__main__":
    main()

