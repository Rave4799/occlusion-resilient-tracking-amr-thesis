#!/usr/bin/env python3
"""Filter extracted train images using timestamp segments and bag metadata."""

import argparse
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
import re
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_INPUT_DIR = str(BASE_DIR / 'dataset' / 'train' / 'images')
DEFAULT_OUTPUT_DIR = str(BASE_DIR / 'dataset' / 'train' / 'images_filtered')
FRAME_RATE = 25  # bag frame rate in frames per second
EXTRACTION_STEP = 3  # images were extracted every 3rd original frame

BAG_SEGMENTS: Dict[str, List[str]] = {
    '2023_05_05 Test1': [
        '00:59-01:19', '01:40-01:48', '05:40-05:50', '07:23-07:30', '08:18-08:34',
        '08:35-08:45', '09:10-09:36', '09:58-10:04', '10:05-10:15', '10:23-10:34',
        '10:57-11:12', '11:18-11:24', '11:50-12:04', '12:22-12:29', '14:19-14:54',
        '15:11-15:45', '16:09-16:20', '17:07-17:24', '17:28-18:12', '18:38-18:46',
        '18:54-19:02', '19:08-19:34',
    ],
    '2023_05_05 Test2': [
        '00:10-00:14', '00:44-00:50', '00:53-01:05', '01:07-01:20', '01:29-01:35',
        '01:37-01:48', '01:50-02:14', '02:25-02:33', '02:51-03:01', '03:08-03:22',
        '03:31-03:38', '04:32-04:48', '04:54-04:56', '05:17-05:23', '07:35-07:36',
        '08:05-08:08', '08:12-08:14', '08:53-09:09', '09:31-09:33', '09:40-09:46',
        '11:18-11:20', '12:06-12:36', '13:33-13:40', '13:42-13:59', '15:34-15:39',
    ],
    '2023_05_05 Test3': [
        '01:06-01:21', '02:16-02:32', '04:10-04:16', '04:48-05:00', '05:10-05:15',
        '05:35-05:38', '05:45-05:53', '06:03-06:09', '06:10-06:16', '06:18-06:30',
        '06:33-06:41', '08:16-08:19', '08:27-08:31', '11:02-11:06', '11:10-11:20',
        '11:22-11:26', '11:43-11:46', '11:52-11:59', '12:26-12:29', '13:14-13:23',
        '13:47-13:51', '13:54-13:59', '14:05-14:10', '14:22-14:26', '14:34-14:37',
        '14:40-14:45', '14:45-14:59', '15:02-15:09', '15:13-15:19', '15:20-15:24',
        '15:28-15:33', '15:35-15:42', '15:45-15:50', '15:53-15:55', '15:55-15:58',
    ],
    '2023_05_05 Test4': [
        '00:12-00:42', '00:52-00:58', '01:00-01:07', '01:10-01:15', '01:17-01:25',
        '01:31-01:36', '01:42-01:50', '02:30-02:43', '02:50-03:02', '03:09-03:12',
        '03:22-03:27', '03:28-03:34', '04:06-04:10', '04:42-04:45', '04:55-05:07',
        '05:49-05:52', '06:08-06:20', '06:23-06:45', '06:55-07:02', '07:09-07:20',
        '08:55-09:05', '09:35-09:46', '10:11-10:14', '10:17-10:20', '10:25-10:40',
        '12:09-12:15', '12:35-12:39', '13:32-13:36', '13:43-13:52',
    ],
    '2023_05_12 Test1': [
        '00:25-00:34', '00:35-00:45', '01:47-01:52', '02:00-02:10', '02:15-02:34',
        '02:45-02:54', '02:56-03:10', '03:20-03:42', '03:55-03:59', '04:55-05:08',
        '05:37-05:40', '06:29-06:39', '06:46-06:55', '06:57-07:05', '07:12-07:19',
        '07:38-07:44', '08:40-08:54', '09:26-09:30', '09:35-09:45', '09:50-10:00',
        '10:05-10:13', '10:15-10:23', '10:45-10:48', '10:53-11:15', '11:35-11:40',
        '11:45-11:57', '12:05-12:14', '13:38-13:49', '16:03-16:14', '16:36-16:39',
        '16:47-16:54',
    ],
    '2023_05_12 Test2': [
        '00:16-00:18', '00:20-00:29', '00:33-00:39', '00:40-00:45', '00:52-01:03',
        '01:05-01:09', '01:10-01:12', '01:14-01:17', '01:22-01:28', '01:30-01:35',
        '01:38-01:41', '01:43-02:04', '02:23-02:28', '02:51-03:02', '03:06-03:11',
        '03:25-03:29', '03:37-03:42', '03:53-03:59', '04:23-04:31', '05:20-05:28',
        '05:39-05:42', '05:54-06:02', '07:42-07:44', '07:56-08:00', '08:05-08:08',
        '08:10-08:15', '08:21-08:27', '08:27-08:31', '09:28-09:32', '10:11-10:15',
        '10:39-10:42', '10:46-11:00', '11:55-12:03', '12:09-12:13', '12:58-13:01',
        '13:10-13:14', '14:07-14:10', '15:58-16:01', '17:38-17:40',
    ],
    '2023_05_12 Test3': [
        '05:40-05:53', '07:43-07:49', '08:18-08:27', '08:28-08:36', '10:23-10:36',
        '11:40-11:45', '11:46-11:52', '11:55-12:00', '12:05-12:15', '12:15-12:20',
        '13:00-13:10', '13:15-13:20', '15:05-15:10', '16:07-16:13',
    ],
    '2023_05_12 Test4': [
        '00:08-00:26', '00:48-01:05', '01:06-01:20', '01:28-01:33', '02:00-02:10',
        '05:55-06:05', '06:45-06:52', '06:55-07:05', '07:30-07:38', '08:03-08:10',
        '09:05-09:10', '09:15-09:25', '09:38-09:42', '10:00-10:06', '10:15-10:36',
        '10:50-11:00', '13:12-13:15', '13:40-14:10',
    ],
    '2023_05_12 Test5': [
        '03:02-03:20', '03:25-03:30', '04:02-04:10', '04:11-04:13', '05:15-05:23',
        '07:58-08:10', '08:25-08:30', '09:05-09:20', '10:40-10:48', '12:17-12:20',
        '12:50-12:54', '13:02-13:07', '13:19-13:25', '13:25-13:35', '14:00-14:10',
        '14:13-14:19', '14:20-14:29',
    ],
    '2023_05_12 Test6': [
        '00:08-02:01', '03:05-03:10', '04:05-04:13', '06:06-06:20', '07:22-07:24',
        '09:12-09:17', '09:25-09:29', '09:40-09:59', '10:33-10:42', '12:14-12:20',
        '12:30-12:48', '14:10-14:25', '14:25-14:50',
    ],
    '2023_05_13 Test1 GH010213': [
        '00:17-00:28', '00:32-00:44', '00:52-01:06', '01:15-01:28', '01:35-01:51',
        '02:35-02:41', '03:12-03:24', '04:10-04:24', '04:55-05:07', '05:07-05:22',
        '05:28-05:38', '05:41-05:52', '06:00-06:15', '06:20-06:32', '07:33-07:36',
        '07:50-08:00', '08:15-08:21', '10:45-10:48', '10:58-11:10', '11:58-12:05',
        '12:52-12:54', '13:33-13:38', '13:40-13:43', '14:02-14:07', '14:08-14:10',
        '14:15-14:20', '14:43-15:00', '15:00-15:15', '15:20-15:30', '15:30-15:39',
        '15:43-15:46',
    ],
    '2023_05_13 Test2 GH010214': [
        '00:24-00:45', '01:05-01:08', '01:13-01:20', '01:28-01:36', '01:45-01:55',
        '02:02-02:25', '02:47-02:52', '02:56-03:10', '03:25-03:55', '04:09-04:11',
        '05:28-05:40', '05:43-05:50', '07:12-07:23', '08:25-08:33', '08:35-08:45',
        '10:05-10:12', '11:27-11:33', '12:05-12:25', '12:40-12:50', '13:45-13:49',
        '13:58-14:04',
    ],
    '2023_05_13 Test3 GH010215': [
        '01:05-01:15', '01:35-02:00', '02:13-02:20', '02:33-02:52', '03:23-03:25',
        '03:27-03:28', '03:45-03:55', '05:46-05:54', '06:30-06:45', '08:33-08:40',
        '09:00-09:04', '09:55-10:02', '10:20-10:36', '11:10-11:25', '11:45-11:57',
        '13:11-13:24', '14:02-14:14', '14:14-14:24', '14:31-14:33', '14:35-14:39',
    ],
}

FILENAME_INDEX_PATTERN = re.compile(r'_(\d{6})\.[^.]+$')
DATE_PREFIX_PATTERN = re.compile(r'^(?P<date>\d{8}|\d{4}-\d{2}-\d{2})_')


def parse_timestamp(timestamp: str) -> float:
    """Convert a timestamp MM:SS to absolute seconds."""
    minutes, seconds = timestamp.split(':')
    return int(minutes) * 60.0 + int(seconds)


def parse_segment(segment: str) -> Tuple[float, float]:
    """Convert a timestamp segment string to a start/end second interval."""
    start_ts, end_ts = segment.split('-')
    start_seconds = parse_timestamp(start_ts)
    end_seconds = parse_timestamp(end_ts)
    if end_seconds < start_seconds:
        raise ValueError(f'Invalid segment range: {segment}')
    return start_seconds, end_seconds


def build_date_segment_map(raw_map: Dict[str, List[str]]) -> Dict[str, List[Tuple[float, float]]]:
    """Build a mapping from bag date keys to numeric second intervals."""
    result: Dict[str, List[Tuple[float, float]]] = {}
    for bag_key, segments in raw_map.items():
        bag_date = bag_key.split()[0].replace('_', '')
        result.setdefault(bag_date, []).extend(parse_segment(segment) for segment in segments)
    return result


def normalize_date_key(date_prefix: str) -> Optional[str]:
    """Normalize filename date prefixes to YYYYMMDD format."""
    if '-' in date_prefix:
        return date_prefix.replace('-', '')
    return date_prefix


def parse_sequence_index(filename: str) -> Optional[int]:
    """Extract the numeric image sequence index from the filename."""
    match = FILENAME_INDEX_PATTERN.search(filename)
    if not match:
        return None
    return int(match.group(1))


def is_time_selected(time_seconds: float, intervals: Iterable[Tuple[float, float]]) -> bool:
    """Determine whether a given time falls inside any of the target intervals."""
    return any(start <= time_seconds <= end for start, end in intervals)


def find_date_key_for_filename(filename: str) -> Optional[str]:
    """Extract the date prefix from an image filename and normalize it."""
    match = DATE_PREFIX_PATTERN.match(filename)
    if not match:
        return None
    return normalize_date_key(match.group('date'))


def collect_image_files(input_dir: str) -> List[str]:
    """Collect JPEG and PNG image file paths from the input directory."""
    images = []
    for item in os.listdir(input_dir):
        lower = item.lower()
        if lower.endswith('.jpg') or lower.endswith('.jpeg') or lower.endswith('.png'):
            images.append(item)
    return sorted(images)


def filter_images(input_dir: str, output_dir: str, dry_run: bool = False) -> Tuple[int, int, int]:
    """Filter the extracted images into the output directory based on bag segment intervals."""
    segments_map = build_date_segment_map(BAG_SEGMENTS)
    images = collect_image_files(input_dir)

    if not images:
        raise FileNotFoundError(f'No image files found in {input_dir}')

    selected_count = 0
    skipped_count = 0
    unmatched_count = 0
    os.makedirs(output_dir, exist_ok=True)

    for image_name in images:
        date_key = find_date_key_for_filename(image_name)
        if date_key is None or date_key not in segments_map:
            unmatched_count += 1
            continue

        seq_index = parse_sequence_index(image_name)
        if seq_index is None:
            skipped_count += 1
            continue

        time_seconds = (seq_index * EXTRACTION_STEP) / FRAME_RATE
        if is_time_selected(time_seconds, segments_map[date_key]):
            selected_count += 1
            src_path = os.path.join(input_dir, image_name)
            dst_path = os.path.join(output_dir, image_name)
            if not dry_run:
                shutil.copy2(src_path, dst_path)
        else:
            skipped_count += 1

    return selected_count, skipped_count, unmatched_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Filter extracted train images using timestamp segments from annotated bags.'
    )
    parser.add_argument(
        '--input-dir', default=DEFAULT_INPUT_DIR,
        help='Path to extracted images to filter (default: %(default)s)'
    )
    parser.add_argument(
        '--output-dir', default=DEFAULT_OUTPUT_DIR,
        help='Path to write selected filtered images (default: %(default)s)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Count selected images without copying files.'
    )
    args = parser.parse_args()

    print('Input directory:', args.input_dir)
    print('Output directory:', args.output_dir)
    print('Dry run mode:' if args.dry_run else 'Executing copy mode')

    selected, skipped, unmatched = filter_images(args.input_dir, args.output_dir, dry_run=args.dry_run)

    print('\nFilter summary:')
    print(f'  Selected images: {selected}')
    print(f'  Skipped images: {skipped}')
    print(f'  Unmatched images: {unmatched}')
    if args.dry_run:
        print('Dry run complete. No files were copied.')
    else:
        print('Filtered images written to:', args.output_dir)


if __name__ == '__main__':
    main()
