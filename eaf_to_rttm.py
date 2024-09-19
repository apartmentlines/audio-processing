#!/usr/bin/env python3

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import Dict, Optional, Iterator


class EAFtoRTTMConverter:
    """
    A class to convert .eaf files produced from the ELAN program into RTTM format.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        debug: bool = False,
    ):
        """
        Initialize the EAFtoRTTMConverter.

        :param input_dir: Directory containing .eaf files to be converted.
        :param output_dir: Directory where the RTTM files will be written.
        :param debug: Flag indicating whether to enable debug logging.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.debug = debug
        self.setup_logging()

    def setup_logging(self):
        """Set up logging configuration."""
        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def run(self):
        """Run the conversion process."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory: {self.output_dir}")

        for eaf_file in self.input_dir.glob("*.eaf"):
            rttm_file = self.output_dir / f"{eaf_file.stem}.rttm"
            logging.info(f"Processing file: {eaf_file}")
            try:
                self.convert_file(eaf_file, rttm_file)
            except Exception as e:
                logging.error(f"Failed to process file {eaf_file}: {e}")

        logging.info("Conversion process completed")

    def convert_file(self, eaf_file: Path, rttm_file: Path):
        """
        Convert a single .eaf file to RTTM format.

        :param eaf_file: Path to the .eaf file.
        :param rttm_file: Path where the RTTM file will be written.
        """
        with rttm_file.open("w") as outfile:
            for annotation in self.parse_eaf_file(eaf_file):
                outfile.write(f"{annotation}\n")
        logging.info(f"Wrote RTTM file: {rttm_file}")

    def parse_eaf_file(self, eaf_file: Path) -> Iterator[str]:
        """
        Parse an .eaf file and yield RTTM annotations.

        :param eaf_file: Path to the .eaf file.
        :yield: RTTM annotation lines.
        """
        try:
            tree = ET.parse(eaf_file)
            root = tree.getroot()

            time_slots = self.build_time_slots(root)
            file_id = self.get_file_id(root)

            for tier in root.findall("TIER"):
                speaker_name = tier.get("TIER_ID")
                logging.debug(f"Processing TIER_ID (speaker): {speaker_name}")

                for annotation in tier.findall("ANNOTATION"):
                    rttm_line = self.process_annotation(
                        annotation, time_slots, file_id, speaker_name
                    )
                    if rttm_line:
                        yield rttm_line

        except ET.ParseError as e:
            logging.error(f"XML parsing error in file {eaf_file}: {e}")
        except Exception as e:
            logging.error(f"Error processing file {eaf_file}: {e}")

    def build_time_slots(self, root: ET.Element) -> Dict[str, int]:
        """
        Build a dictionary of time slots from the XML root.

        :param root: XML root element.
        :return: Dictionary mapping TIME_SLOT_ID to TIME_VALUE.
        """
        time_slots = {}
        time_order = root.find("TIME_ORDER")
        for time_slot in time_order.findall("TIME_SLOT"):
            time_slot_id = time_slot.get("TIME_SLOT_ID")
            time_value = time_slot.get("TIME_VALUE")
            if time_value is None:
                logging.error(f"Missing TIME_VALUE for TIME_SLOT_ID {time_slot_id}")
                continue
            try:
                time_slots[time_slot_id] = int(time_value)
                logging.debug(f"Time slot {time_slot_id} = {time_value} ms")
            except ValueError:
                logging.error(
                    f"Invalid TIME_VALUE '{time_value}' for TIME_SLOT_ID {time_slot_id}"
                )
        return time_slots

    def get_file_id(self, root: ET.Element) -> str:
        """
        Extract the file ID from the XML root.

        :param root: XML root element.
        :return: File ID string.
        """
        header = root.find("HEADER")
        media_descriptor = header.find("MEDIA_DESCRIPTOR")
        media_url = media_descriptor.get("MEDIA_URL")
        logging.debug(f"Media URL: {media_url}")

        parsed_url = urlparse(media_url)
        media_file_path = unquote(parsed_url.path)
        logging.debug(f"Media file path: {media_file_path}")

        file_id = Path(media_file_path).stem
        logging.debug(f"File ID: {file_id}")
        return file_id

    def process_annotation(
        self,
        annotation: ET.Element,
        time_slots: Dict[str, int],
        file_id: str,
        speaker_name: str,
    ) -> Optional[str]:
        """
        Process a single annotation and return an RTTM line.

        :param annotation: XML element representing an annotation.
        :param time_slots: Dictionary of time slots.
        :param file_id: File ID string.
        :param speaker_name: Speaker name string (derived from TIER_ID in the EAF file).
        :return: RTTM line string or None if invalid.
        """
        alignable_annotation = annotation.find("ALIGNABLE_ANNOTATION")
        if alignable_annotation is None:
            logging.warning(
                f"No ALIGNABLE_ANNOTATION found under ANNOTATION in TIER {speaker_name}"
            )
            return None

        annotation_id = alignable_annotation.get("ANNOTATION_ID")
        time_slot_ref1 = alignable_annotation.get("TIME_SLOT_REF1")
        time_slot_ref2 = alignable_annotation.get("TIME_SLOT_REF2")

        if time_slot_ref1 not in time_slots or time_slot_ref2 not in time_slots:
            logging.error(
                f"Time slot reference not found for annotation {annotation_id}"
            )
            return None

        start_time_ms = time_slots[time_slot_ref1]
        end_time_ms = time_slots[time_slot_ref2]
        onset_sec = start_time_ms / 1000.0
        offset_sec = end_time_ms / 1000.0
        duration_sec = offset_sec - onset_sec

        if duration_sec < 0:
            logging.error(f"Negative duration for annotation {annotation_id}")
            return None

        # Check for non-empty ANNOTATION_VALUE
        # Note: In this format, we expect ANNOTATION_VALUE to be empty.
        # We only log when it's non-empty as it's an unusual case.
        annotation_value = alignable_annotation.find("ANNOTATION_VALUE")
        if annotation_value is not None and annotation_value.text:
            logging.warning(
                f"Non-empty ANNOTATION_VALUE found for annotation {annotation_id}: {annotation_value.text}"
            )

        logging.debug(
            f"Annotation {annotation_id}: onset {onset_sec} s, duration {duration_sec} s, speaker {speaker_name}"
        )
        # Channel ID is always set to "1" as per RTTM specification
        return f"SPEAKER {file_id} 1 {onset_sec:.3f} {duration_sec:.3f} <NA> <NA> {speaker_name} <NA> <NA>"


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="EAF to RTTM Converter: Convert .eaf files produced from the ELAN program into RTTM format files.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("eaf"),
        help="Directory containing .eaf files to be converted (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rttm"),
        help="Directory to write RTTM files (default: %(default)s).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for more detailed output.",
    )
    return parser.parse_args()


def main():
    """
    Main function to parse command-line arguments and initiate the conversion process.
    """
    args = parse_arguments()
    try:
        converter = EAFtoRTTMConverter(
            input_dir=args.input_dir, output_dir=args.output_dir, debug=args.debug
        )
        converter.run()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
