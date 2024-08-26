#!/bin/bash

# Start with docker command:
# docker run --gpus all -it --entrypoint /bin/bash  -v "$(pwd):/app"   custom-whisperx
# DON'T FORGET: export HUGGINGFACEHUB_API_TOKEN=hf_...

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
  local total_files=$(find "${audio_dir}" -type f -name "*.wav" | wc -l)
  local processed=0
  local succeeded=0

  echo "Found ${total_files} .wav files to process."

  find "${audio_dir}" -type f -name "*.wav" | while read -r audio_file; do
    ((processed++))
    echo "Processing file ${processed}/${total_files}: ${audio_file}"

    subfolder=$(echo "${audio_file}" | cut -d'/' -f2)
    base_name=$(basename "${audio_file}" .wav)

    transcribe "${audio_file}"

    if [ $? -eq 0 ]; then
      srt_file="${base_name}.srt"
      transcription_dir="transcriptions/${subfolder}"
      mkdir -p "${transcription_dir}"
      mv -v "${srt_file}" "${transcription_dir}/"
      ((succeeded++))
      echo "Transcription successful. Moved ${srt_file} to ${transcription_dir}/"
    else
      echo "Transcription failed for ${audio_file}"
    fi

    echo "Completed ${processed}/${total_files} files."
  done

  echo "All files processed. Total successful transcriptions: ${succeeded} out of ${processed} processed, ${total_files} total."
}

get_audio_files
