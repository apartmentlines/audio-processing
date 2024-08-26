#!/bin/bash

# Start with docker command:
# docker run --gpus all -it --entrypoint /bin/bash  -v "$(pwd):/app"   custom-whisperx
# DON'T FORGET: export HUGGINGFACEHUB_API_TOKEN=hf_...

PROCESSED=0
SUCCEEDED=0

function transcribe() {
  local input_file="${1}"
  echo "Transcribing: ${input_file}"
  whisperx \
    --model "${WHISPER_MODEL}" \
    --language "${LANG}" \
    --diarize \
    --min_speakers 2 \
    --max_speakers 2 \
    --hf_token "${HUGGINGFACEHUB_API_TOKEN}"\
    --output_format srt \
    "${input_file}"
}

function get_audio_files() {
  local audio_dir="audio"
  local files_list="$(find "${audio_dir}" -type f -name "*.wav")"
  local total_files=$(echo "${files_list}" | wc -l)

  echo "Found ${total_files} .wav files to process."

  for audio_file in ${files_list}; do
    ((PROCESSED++))
    echo "Processing file ${PROCESSED}/${total_files}: ${audio_file}"

    subfolder=$(echo "${audio_file}" | cut -d'/' -f2)
    base_name=$(basename "${audio_file}" .wav)

    transcribe "${audio_file}" > /dev/null 2>&1

    if [ $? -eq 0 ]; then
      srt_file="${base_name}.srt"
      transcription_dir="transcriptions/${subfolder}"
      mkdir -p "${transcription_dir}"
      mv "${srt_file}" "${transcription_dir}/"
      ((SUCCEEDED++))
      echo "Transcription successful. Moved ${srt_file} to ${transcription_dir}/"
    else
      echo "Transcription failed for ${audio_file}"
    fi

    echo "Completed ${PROCESSED}/${total_files} files."
  done

  echo "All files processed. Total successful transcriptions: ${SUCCEEDED} out of ${PROCESSED} processed, ${total_files} total."
}

function calculate_time_stats() {
  local start_time=$1
  local end_time=$2

  local total_seconds=$((end_time - start_time))
  local hours=$((total_seconds / 3600))
  local minutes=$(( (total_seconds % 3600) / 60 ))
  local seconds=$((total_seconds % 60))

  local avg_seconds_per_transcription=0
  if [ ${SUCCEEDED} -gt 0 ]; then
    avg_seconds_per_transcription=$((total_seconds / SUCCEEDED))
    local remainder=$((total_seconds % SUCCEEDED))
    remainder=$((remainder * 100 / SUCCEEDED))
  fi

  echo "Total time: ${hours}h ${minutes}m ${seconds}s"
  echo "Average time per transcription: ${avg_seconds_per_transcription}.${remainder} seconds"
}

# Capture start time
start_time=$(date +%s)

get_audio_files

# Capture end time
end_time=$(date +%s)

# Calculate and display time stats
calculate_time_stats "${start_time}" "${end_time}"
