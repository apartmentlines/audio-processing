import subprocess
import json
from pydub import AudioSegment
import srt


def split_stereo(input_file, left_output, right_output):
    audio = AudioSegment.from_file(input_file)
    left_channel = audio.split_to_mono()[0]
    right_channel = audio.split_to_mono()[1]
    left_channel.export(left_output, format="wav")
    right_channel.export(right_output, format="wav")


def transcribe_with_whisperx(audio_file):
    command = f"whisperx {audio_file} --output_format json"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def merge_transcriptions(left_trans, right_trans, left_label, right_label):
    merged = []
    for segment in left_trans["segments"]:
        merged.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": f"{left_label}: {segment['text']}",
                "channel": "left",
            }
        )
    for segment in right_trans["segments"]:
        merged.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": f"{right_label}: {segment['text']}",
                "channel": "right",
            }
        )
    return sorted(merged, key=lambda x: x["start"])


def create_srt(merged_trans, output_file):
    subs = []
    for i, segment in enumerate(merged_trans, start=1):
        sub = srt.Subtitle(
            index=i,
            start=srt.timedelta(seconds=segment["start"]),
            end=srt.timedelta(seconds=segment["end"]),
            content=segment["text"],
        )
        subs.append(sub)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))


# Main execution
input_file = "conversation.wav"
left_output = "left_channel.wav"
right_output = "right_channel.wav"
left_label = "Speaker A"
right_label = "Speaker B"
output_srt = "transcription.srt"

split_stereo(input_file, left_output, right_output)

left_trans = transcribe_with_whisperx(left_output)
right_trans = transcribe_with_whisperx(right_output)

merged_trans = merge_transcriptions(left_trans, right_trans, left_label, right_label)

create_srt(merged_trans, output_srt)
