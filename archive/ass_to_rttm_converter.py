import argparse
import logging
import re
from dataclasses import dataclass
from typing import List

@dataclass
class Subtitle:
    start: float
    end: float
    speaker: str
    text: str

class ASStoRTTMConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse_time(self, time_str: str) -> float:
        """Convert ASS time format to seconds."""
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    def parse_ass(self, ass_content: str) -> List[Subtitle]:
        """Parse ASS content and extract relevant subtitle information."""
        subtitles = []
        in_events_section = False
        events_format = []

        for line in ass_content.split('\n'):
            line = line.strip()
            if line == '[Events]':
                in_events_section = True
                continue
            if not in_events_section:
                continue
            if line.startswith('Format:'):
                events_format = [f.strip() for f in line.split(':', 1)[1].split(',')]
                continue
            if line.startswith('Dialogue:'):
                # Split the line, but limit the split to the number of fields minus one
                # This ensures that commas in the Text field are preserved
                fields = line.split(',', len(events_format) - 1)

                if len(fields) != len(events_format):
                    self.logger.warning(f"Mismatched field count in line: {line}")
                    continue

                # Create a dictionary mapping format fields to values
                event_dict = dict(zip(events_format, fields))

                # Extract relevant information
                start = self.parse_time(event_dict['Start'])
                end = self.parse_time(event_dict['End'])
                text = event_dict['Text'].strip()

                # Extract speaker from text
                speaker_match = re.match(r'\s*\[\s*([^\]]+)\s*\]\s*:\s*(.+)', text)
                if speaker_match:
                    speaker = speaker_match.group(1).strip()
                    text = speaker_match.group(2).strip()
                else:
                    speaker = "UNKNOWN"

                subtitles.append(Subtitle(start=start, end=end, speaker=speaker, text=text))

        self.logger.info(f"Parsed {len(subtitles)} subtitles from ASS content")
        return subtitles

    def merge_adjacent_segments(self, subtitles: List[Subtitle], max_gap: float = 0.3) -> List[Subtitle]:
        """
        Merge adjacent segments from the same speaker if the gap is less than max_gap seconds.
        Default max_gap is 0.3 seconds (300 ms).
        """
        merged = []
        for sub in subtitles:
            if merged and merged[-1].speaker == sub.speaker and (sub.start - merged[-1].end) < max_gap:
                merged[-1].end = sub.end
                merged[-1].text += ' ' + sub.text
            else:
                merged.append(sub)

        self.logger.info(f"Merged subtitles into {len(merged)} segments")
        return merged

    def generate_rttm(self, subtitles: List[Subtitle]) -> str:
        """Generate RTTM format string from subtitles."""
        rttm_lines = []
        for sub in subtitles:
            duration = sub.end - sub.start
            rttm_lines.append(f"SPEAKER unknown 1 {sub.start:.3f} {duration:.3f} <NA> <NA> {sub.speaker} <NA> <NA>")

        self.logger.info(f"Generated {len(rttm_lines)} RTTM lines")
        return '\n'.join(rttm_lines)

    def convert(self, ass_content: str) -> str:
        """Convert ASS content to RTTM format."""
        subtitles = self.parse_ass(ass_content)
        merged_subtitles = self.merge_adjacent_segments(subtitles)
        return self.generate_rttm(merged_subtitles)

def main():
    parser = argparse.ArgumentParser(description="Convert ASS subtitle format to RTTM format")
    parser.add_argument('-i', '--input', default='input.ass', help="Input ASS file path")
    parser.add_argument('-o', '--output', default='output.rttm', help="Output RTTM file path")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    converter = ASStoRTTMConverter()

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            ass_content = f.read()

        logger.info(f"Successfully read input file: {args.input}")

        rttm_content = converter.convert(ass_content)

        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(rttm_content)

        logger.info(f"Successfully wrote output file: {args.output}")

    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input}")
    except IOError as e:
        logger.error(f"IO error occurred: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
