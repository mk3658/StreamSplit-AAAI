"""
Download and prepare FULL AudioSet dataset.
Downloads REAL audio from YouTube using yt-dlp.
"""

import argparse
import yaml
import sys
import os
import subprocess
import pandas as pd
import urllib.request
import json
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_yt_dlp():
    """Check if yt-dlp is installed and install if needed."""
    try:
        result = subprocess.run(
            ['yt-dlp', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✅ yt-dlp version: {result.stdout.strip()}")
            return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print("⚠️ yt-dlp not found. Installing...")
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', 'yt-dlp'],
            check=True,
            timeout=60
        )
        print("✅ yt-dlp installed successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to install yt-dlp: {e}")
        return False


def download_youtube_audio(args_tuple):
    """
    Download audio segment from YouTube video using yt-dlp.
    
    Args:
        args_tuple: (video_id, output_path, start_time, duration, index, total)
    
    Returns:
        tuple: (success: bool, video_id: str, filepath: str)
    """
    video_id, output_path, start_time, duration, index, total = args_tuple
    
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        output_template = os.path.join(output_path, f"{video_id}_{int(start_time)}.%(ext)s")
        output_file = os.path.join(output_path, f"{video_id}_{int(start_time)}.wav")
        
        # Skip if already exists
        if os.path.exists(output_file):
            return (True, video_id, output_file)
        
        # yt-dlp command with audio extraction
        cmd = [
            'yt-dlp',
            '-f', 'bestaudio/best',
            '--extract-audio',
            '--audio-format', 'wav',
            '--audio-quality', '0',
            '--postprocessor-args', f'ffmpeg:-ss {start_time} -t {duration} -ar 16000 -ac 1',
            '-o', output_template,
            '--no-playlist',
            '--quiet',
            '--no-warnings',
            '--ignore-errors',
            '--no-check-certificate',
            url
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60,
            text=True
        )
        
        # Check if file was created
        if os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
            return (True, video_id, output_file)
        else:
            return (False, video_id, None)
            
    except subprocess.TimeoutExpired:
        return (False, video_id, None)
    except Exception as e:
        return (False, video_id, None)


def download_audioset_metadata(csv_path, subset='balanced'):
    """Download AudioSet metadata CSV if not exists."""
    if os.path.exists(csv_path):
        print(f"✅ Metadata already exists at {csv_path}")
        return True
    
    print(f"Downloading AudioSet {subset} metadata CSV...")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # AudioSet CSV URLs
    urls = {
        'balanced': "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv",
        'eval': "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv",
        'unbalanced': "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv"
    }
    
    try:
        url = urls.get(subset, urls['balanced'])
        urllib.request.urlretrieve(url, csv_path)
        print(f"✅ Downloaded metadata to {csv_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download metadata: {e}")
        return False


def download_audioset_real(csv_path, output_dir, num_samples=100, num_workers=2, duration=10):
    """
    Download real AudioSet data from YouTube.
    
    Args:
        csv_path: Path to AudioSet CSV metadata
        output_dir: Output directory for audio files
        num_samples: Number of samples to download
        num_workers: Number of parallel download workers
        duration: Audio clip duration in seconds
    
    Returns:
        int: Number of successfully downloaded samples
    """
    print(f"\n{'='*60}")
    print("🌐 DOWNLOADING REAL AUDIOSET FROM YOUTUBE")
    print(f"{'='*60}")
    
    # Read CSV
    print(f"\n📄 Reading AudioSet metadata from {csv_path}...")
    try:
        # AudioSet CSV format: skip first 3 comment lines
        df = pd.read_csv(csv_path, skiprows=3, sep=', ', engine='python')
        df.columns = ['YTID', 'start_seconds', 'end_seconds', 'positive_labels']
        print(f"✅ Found {len(df)} total videos in metadata")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return 0
    
    # Limit samples
    df = df.head(num_samples)
    print(f"📊 Attempting to download {len(df)} samples...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments for parallel download
    download_args = [
        (row['YTID'], output_dir, float(row['start_seconds']), duration, idx, len(df))
        for idx, (_, row) in enumerate(df.iterrows(), 1)
    ]
    
    # Download in parallel with progress bar
    print(f"\n⬇️  Downloading with {num_workers} parallel workers...")
    print("⏳ This may take 5-15 minutes depending on your internet speed...\n")
    
    successful_downloads = []
    failed_downloads = []
    
    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(download_args), desc="Downloading", unit="file") as pbar:
            for result in pool.imap_unordered(download_youtube_audio, download_args):
                success, video_id, filepath = result
                if success:
                    successful_downloads.append((video_id, filepath))
                else:
                    failed_downloads.append(video_id)
                pbar.update(1)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Successfully downloaded: {len(successful_downloads)}/{len(df)} samples")
    print(f"❌ Failed downloads: {len(failed_downloads)}")
    print(f"{'='*60}")
    
    if failed_downloads and len(failed_downloads) <= 10:
        print(f"\n⚠️  Failed video IDs: {', '.join(failed_downloads[:10])}")
    
    return len(successful_downloads)


def main():
    parser = argparse.ArgumentParser(
        description='Download FULL AudioSet dataset from YouTube'
    )
    parser.add_argument('--config', type=str,
                       default='configs/streamsplit.yaml',
                       help='Path to configuration file')
    parser.add_argument('--subset', type=str, default='balanced',
                       choices=['balanced', 'eval', 'unbalanced', 'all'],
                       help='AudioSet subset to download (use "all" for complete dataset)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Limit number of samples (default: None = download all)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of parallel download workers (default: 4)')
    parser.add_argument('--duration', type=int, default=10,
                       help='Audio clip duration in seconds')
    parser.add_argument('--csv_path', type=str, default=None,
                       help='Path to AudioSet CSV metadata (auto-download if not provided)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for audio files (default: data/audioset/audio)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume previous download (skip existing files)')
    parser.add_argument('--skip_failed', action='store_true',
                       help='Skip failed downloads from previous run')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("🌐 AUDIOSET DOWNLOADER - REAL YOUTUBE AUDIO")
    print(f"{'='*60}\n")
    
    # Check/install yt-dlp
    if not check_yt_dlp():
        print("❌ Cannot proceed without yt-dlp.")
        print("Please install it manually: pip install yt-dlp")
        return 1
    
    # Determine output directory
    if args.output_dir is None:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            data_dir = config['data']['data_dir']
        else:
            data_dir = './data'
        args.output_dir = os.path.join(data_dir, 'audioset', 'audio')
    
    print(f"Configuration:")
    print(f"  Subset: {args.subset}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Duration: {args.duration}s")
    print(f"  Resume: {args.resume}")
    
    # Handle "all" subset - download all three subsets
    if args.subset == 'all':
        subsets = ['balanced', 'eval', 'unbalanced']
        print(f"\n📦 Downloading ALL AudioSet subsets: {', '.join(subsets)}")
        print("⚠️  This will download ~2 million audio clips (may take days)")
        print("⚠️  Requires ~400-600 GB of storage space")
        
        response = input("\nContinue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Download cancelled.")
            return 0
        
        total_downloaded = 0
        for subset in subsets:
            print(f"\n{'='*60}")
            print(f"Processing subset: {subset}")
            print(f"{'='*60}")
            
            # Determine CSV path for this subset
            csv_path = os.path.join(
                os.path.dirname(args.output_dir),
                f'{subset}_train_segments.csv'
            )
            
            # Download metadata
            if not download_audioset_metadata(csv_path, subset):
                print(f"❌ Failed to download {subset} metadata. Skipping...")
                continue
            
            # Download audio
            num_downloaded = download_audioset_real(
                csv_path,
                args.output_dir,
                num_samples=args.num_samples,
                num_workers=args.num_workers,
                duration=args.duration
            )
            total_downloaded += num_downloaded
        
        print(f"\n{'='*60}")
        print(f"✅ TOTAL DOWNLOADED: {total_downloaded} audio files")
        print(f"{'='*60}")
    
    else:
        # Download single subset
        # Determine CSV path
        if args.csv_path is None:
            args.csv_path = os.path.join(
                os.path.dirname(args.output_dir),
                f'{args.subset}_train_segments.csv'
            )
        
        # Download metadata
        print(f"\n📄 Downloading AudioSet metadata...")
        if not download_audioset_metadata(args.csv_path, args.subset):
            print("❌ Cannot download metadata.")
            return 1
        
        # Download real audio from YouTube
        print(f"\n⬇️  Downloading audio files...")
        num_downloaded = download_audioset_real(
            args.csv_path,
            args.output_dir,
            num_samples=args.num_samples,
            num_workers=args.num_workers,
            duration=args.duration
        )
        
        if num_downloaded > 0:
            print(f"\n{'='*60}")
            print(f"✅ Successfully downloaded {num_downloaded} audio files!")
            print(f"📁 Saved to: {args.output_dir}")
            print(f"{'='*60}")
        else:
            print("\n❌ No samples downloaded.")
            return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
