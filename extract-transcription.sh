date && \
  docker run \
  --rm \
  --gpus all -it \
  -v "$(pwd):/app" \
  custom-whisperx -- \
  --diarize \
  --min_speakers 2 --max_speakers 2 \
  --hf_token ${HUGGINGFACEHUB_API_TOKEN} \
  --output_format srt \
  "${1}" && \
  date && \
  sudo chown -v hunmonk:hunmonk *.srt && \
  mv -v *.srt transcriptions/
