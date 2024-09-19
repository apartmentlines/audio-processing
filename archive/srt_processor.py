#!/usr/bin/env python3

import argparse
import logging
import re
import json
from typing import List, Dict, Union, Optional

MISSING_SPEAKER_DEFAULT = "Unknown"


class SRTProcessor:
    """
    A class for processing SRT files.
    """

    def __init__(self, srt_file: str):
        """
        Initialize the SRTProcessor.

        :param srt_file: Path to the SRT file
        """
        self.srt_file = srt_file
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Initializing SRTProcessor with file: {srt_file}")
        self.content = self._read_srt_file()

    def _read_srt_file(self) -> str:
        """
        Read the content of the SRT file and perform basic validation.

        :return: Content of the SRT file as a string
        :raises FileNotFoundError: If the SRT file is not found
        :raises ValueError: If the SRT file format is invalid
        """
        try:
            with open(self.srt_file, "r", encoding="utf-8") as file:
                content = file.read()

            self.logger.debug(
                f"Successfully read {len(content)} characters from {self.srt_file}"
            )

            # Basic validation of SRT format
            blocks = content.strip().split("\n\n")
            self.logger.debug(f"Found {len(blocks)} blocks in the SRT file")
            for i, block in enumerate(blocks, 1):
                if not self._validate_block(block, i):
                    raise ValueError(f"Invalid SRT format in block {i}")

            return content
        except FileNotFoundError:
            self.logger.error(f"SRT file not found: {self.srt_file}")
            raise
        except ValueError as e:
            self.logger.error(f"Invalid SRT file format: {e}")
            raise

    def _validate_block(self, block: str, block_number: int) -> bool:
        """
        Validate an individual SRT block.

        :param block: The SRT block to validate
        :param block_number: The number of the block in the file
        :return: True if the block is valid, False otherwise
        """
        lines = block.split("\n")
        self.logger.debug(f"Validating block {block_number} with {len(lines)} lines")
        if len(lines) < 3:
            self.logger.warning(f"Block {block_number} has fewer than 3 lines")
            return False

        # Check if the first line is a number
        if not lines[0].isdigit():
            self.logger.warning(f"Block {block_number} does not start with a number")
            return False

        # Check if the second line contains a timestamp
        timestamp_pattern = r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}"
        if not re.match(timestamp_pattern, lines[1]):
            self.logger.warning(f"Block {block_number} has an invalid timestamp format")
            return False

        self.logger.debug(f"Block {block_number} is valid")
        return True

    def find_missing_speaker_tags(self) -> List[int]:
        """
        Find block numbers where speaker tags are missing.

        :return: List of block numbers with missing speaker tags
        """
        missing_tags = []
        blocks = self.content.split("\n\n")
        self.logger.debug(f"Searching for missing speaker tags in {len(blocks)} blocks")
        for i, block in enumerate(blocks, 1):
            lines = block.split("\n")
            if len(lines) > 2 and not re.match(r"\[SPEAKER_\d+\]:", lines[2]):
                missing_tags.append(i)
                self.logger.debug(f"Missing speaker tag in block {i}")

        self.logger.info(f"Found {len(missing_tags)} blocks with missing speaker tags")
        return missing_tags

    def update_speaker_tags(self, updates: Dict[int, str]) -> str:
        """
        Update speaker tags for specified block numbers.

        :param updates: Dictionary of block numbers and new speaker labels
        :return: Updated SRT content as a string
        """
        blocks = self.content.split("\n\n")
        updated_blocks = []
        self.logger.debug(f"Updating speaker tags for {len(updates)} blocks")

        pattern = r"^(\d+\n[\d:,]+ --> [\d:,]+\n)(?:\[SPEAKER_\d+\]:)?\s*(.*(?:\n.*)*)"

        for i, block in enumerate(blocks, 1):
            if i in updates:
                match = re.match(pattern, block, re.MULTILINE | re.DOTALL)
                if match:
                    header, content = match.groups()
                    new_label = updates[i]
                    updated_block = f"{header}[{new_label}]: {content.strip()}"
                    updated_blocks.append(updated_block)
                    self.logger.debug(
                        f"Updated speaker tag in block {i} to [{new_label}]"
                    )
                else:
                    self.logger.warning(f"Block {i} does not match expected format")
                    updated_blocks.append(block)
            else:
                updated_blocks.append(block)

        invalid_blocks = set(updates.keys()) - set(range(1, len(blocks) + 1))
        if invalid_blocks:
            self.logger.error(
                f"The following block numbers are out of range and were ignored: {sorted(invalid_blocks)}"
            )

        return "\n\n".join(updated_blocks)

    def reformat_to_markdown(
        self, speaker_substitutions: Dict[str, str], output_file: Optional[str] = None
    ) -> Union[str, bool]:
        """
        Reformat the SRT content to human-readable markdown.

        :param speaker_substitutions: Dictionary of speaker substitutions
        :param output_file: Optional output file path
        :return: Markdown content as a string if output_file is None, else boolean indicating success
        """
        blocks = self.content.split("\n\n")
        markdown_lines = []
        current_speaker = None
        self.logger.debug(f"Reformatting {len(blocks)} blocks to markdown")

        pattern = r"(?:\[SPEAKER_(\d+)\]:)?\s*(.*)"

        for i, block in enumerate(blocks, 1):
            self.logger.debug(f"Processing block {i}")
            lines = block.split("\n")
            self.logger.debug(f"Block {i} has {len(lines)} lines")
            if len(lines) > 2:
                self.logger.debug(f"Attempting to match pattern in block {i}")
                match = re.match(pattern, lines[2])
                if match:
                    speaker_num, text = match.groups()
                    self.logger.debug(
                        f"Block {i} matched. Speaker number: {speaker_num}, Text: {text[:20]}..."
                    )
                    if speaker_num:
                        speaker = speaker_substitutions.get(
                            f"SPEAKER_{speaker_num}", MISSING_SPEAKER_DEFAULT
                        )
                        self.logger.debug(
                            f"Speaker {speaker_num} substituted with {speaker}"
                        )
                    else:
                        speaker = MISSING_SPEAKER_DEFAULT
                        self.logger.debug(
                            f"No speaker number found, using default: {MISSING_SPEAKER_DEFAULT}"
                        )

                    text = text.strip()
                    self.logger.debug(f"Stripped text in block {i}: {text[:20]}...")

                    if speaker != current_speaker:
                        if current_speaker is not None:
                            markdown_lines.append(
                                ""
                            )  # Add an empty line between speakers
                        markdown_lines.append(f"### {speaker}")
                        current_speaker = speaker
                        self.logger.debug(f"New speaker in block {i}: {speaker}")

                    markdown_lines.append(f"{text}  ")  # Add two spaces for line break
                    self.logger.debug(f"Added text for block {i}")
                else:
                    self.logger.warning(f"Block {i} does not match expected format")
            else:
                self.logger.warning(f"Block {i} has fewer than 3 lines, skipping")

        markdown_content = "\n".join(markdown_lines).strip()
        self.logger.debug(
            f"Generated {len(markdown_content)} characters of markdown content"
        )

        if output_file:
            try:
                with open(output_file, "w", encoding="utf-8") as file:
                    file.write(markdown_content)
                self.logger.info(f"Markdown content written to {output_file}")
                return True
            except IOError:
                self.logger.error(f"Failed to write markdown content to {output_file}")
                return False
        else:
            return markdown_content


def parse_speaker_substitutions(substitutions_str: str) -> Dict[int, str]:
    """
    Parse the speaker substitutions string into a dictionary.

    :param substitutions_str: String of speaker substitutions
    :return: Dictionary of block numbers and new speaker labels
    """
    substitutions = {}
    pairs = substitutions_str.split(",")
    for pair in pairs:
        block_num, label = pair.strip().split(":")
        substitutions[int(block_num.strip())] = label.strip()
    return substitutions


def main():
    parser = argparse.ArgumentParser(description="Process SRT files")
    parser.add_argument("srt_file", help="Path to the SRT file")
    parser.add_argument(
        "--find-missing",
        action="store_true",
        help="Find blocks with missing speaker tags",
    )
    parser.add_argument(
        "--update-speakers",
        help="Update speaker tags (format: 'block:label,block:label')",
    )
    parser.add_argument("--reformat", action="store_true", help="Reformat to markdown")
    parser.add_argument("--speaker-00", help="Substitution for SPEAKER_00")
    parser.add_argument("--speaker-01", help="Substitution for SPEAKER_01")
    parser.add_argument("--output", help="Output file for markdown reformatting")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)
    logger.debug("Debug logging enabled")

    processor = SRTProcessor(args.srt_file)

    if args.find_missing:
        missing_tags = processor.find_missing_speaker_tags()
        print(f"Blocks with missing speaker tags: {missing_tags}")

    if args.update_speakers:
        updates = parse_speaker_substitutions(args.update_speakers)
        updated_content = processor.update_speaker_tags(updates)
        print(updated_content)

    if args.reformat:
        if not (args.speaker_00 and args.speaker_01):
            parser.error("--speaker-00 and --speaker-01 are required for reformatting")
        speaker_substitutions = {
            "SPEAKER_00": args.speaker_00,
            "SPEAKER_01": args.speaker_01,
        }
        result = processor.reformat_to_markdown(speaker_substitutions, args.output)
        if args.output:
            print(f"Reformatting {'successful' if result else 'failed'}")
        else:
            print(result)


if __name__ == "__main__":
    main()
