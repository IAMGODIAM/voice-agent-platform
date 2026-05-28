#!/usr/bin/env python3
"""
YouTube Voice Extraction Pipeline
Extracts and processes voice samples from YouTube videos/playlists
for CSM-1B fine-tuning.

Usage:
    python3 youtube_voice_extract.py "https://www.youtube.com/playlist?list=XXXXX" --output-dir ./voice_data
    python3 youtube_voice_extract.py "https://www.youtube.com/watch?v=XXXXX" --output-dir ./voice_data
"""

import os
import sys
import json
import subprocess
import argparse
import re
from pathlib import Path
from datetime import datetime

class YouTubeVoiceExtractor:
    """
    Extracts voice samples from YouTube videos.
    
    Pipeline:
    1. Download audio from YouTube (yt-dlp)
    2. Convert to 16kHz WAV (ffmpeg)
    3. Voice Activity Detection → segment speech
    4. Quality filter (duration, SNR)
    5. Output: segmented WAV files + metadata JSON
    """
    
    def __init__(self, output_dir: str, sample_rate: int = 16000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.segments_dir = self.output_dir / "segments"
        self.segments_dir.mkdir(exist_ok=True)
        self.metadata = {
            "created": datetime.now().isoformat(),
            "source": None,
            "videos": [],
            "total_segments": 0,
            "total_duration_seconds": 0,
        }
    
    def extract_from_playlist(self, playlist_url: str, max_videos: int = None):
        """Extract audio from all videos in a YouTube playlist."""
        print(f"[YouTube Voice] Extracting from playlist: {playlist_url}")
        
        # Get playlist video list
        result = subprocess.run(
            ["yt-dlp", "--flat-playlist", "--print", "id", playlist_url],
            capture_output=True, text=True, timeout=120
        )
        
        if result.returncode != 0:
            print(f"[ERROR] Failed to get playlist: {result.stderr[:200]}")
            return []
        
        video_ids = [v.strip() for v in result.stdout.strip().split('\n') if v.strip()]
        if max_videos:
            video_ids = video_ids[:max_videos]
        
        print(f"[YouTube Voice] Found {len(video_ids)} videos in playlist")
        
        all_segments = []
        for i, vid in enumerate(video_ids):
            url = f"https://www.youtube.com/watch?v={vid}"
            print(f"\n[YouTube Voice] Processing video {i+1}/{len(video_ids)}: {vid}")
            segments = self.extract_from_video(url, video_index=i)
            all_segments.extend(segments)
        
        # Save metadata
        self.metadata["total_segments"] = len(all_segments)
        self.metadata["source"] = playlist_url
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"\n[YouTube Voice] COMPLETE: {len(all_segments)} segments extracted")
        print(f"[YouTube Voice] Output: {self.output_dir}")
        
        return all_segments
    
    def extract_from_video(self, video_url: str, video_index: int = 0) -> list:
        """Extract voice segments from a single YouTube video."""
        video_id = self._extract_video_id(video_url)
        raw_audio = self.output_dir / f"raw_{video_id}.wav"
        segments = []
        
        try:
            # Step 1: Download audio
            print(f"  [1/4] Downloading audio...")
            self._download_audio(video_url, raw_audio)
            
            if not raw_audio.exists():
                print(f"  [ERROR] Download failed for {video_id}")
                return segments
            
            # Step 2: Get video metadata
            video_meta = self._get_video_metadata(video_url)
            print(f"  Video: {video_meta.get('title', 'Unknown')[:60]}")
            
            # Step 3: Segment by voice activity
            print(f"  [2/4] Segmenting voice activity...")
            segments = self._segment_speech(raw_audio, video_id, video_index)
            
            # Step 4: Quality filter
            print(f"  [3/4] Quality filtering ({len(segments)} raw segments)...")
            segments = self._quality_filter(segments)
            
            # Step 5: Save metadata
            print(f"  [4/4] Saving {len(segments)} clean segments...")
            self.metadata["videos"].append({
                "id": video_id,
                "url": video_url,
                "title": video_meta.get("title", ""),
                "duration": video_meta.get("duration", 0),
                "segments": len(segments),
            })
            
            total_dur = sum(s["duration"] for s in segments)
            self.metadata["total_duration_seconds"] += total_dur
            print(f"  Result: {len(segments)} segments, {total_dur:.1f}s total")
            
        except Exception as e:
            print(f"  [ERROR] Processing failed: {e}")
        finally:
            # Clean up raw audio to save space
            if raw_audio.exists():
                raw_audio.unlink()
        
        return segments
    
    def _download_audio(self, url: str, output_path: Path):
        """Download audio from YouTube as WAV."""
        cmd = [
            "yt-dlp",
            "--no-download-archive",
            "-x",                          # Extract audio only
            "--audio-format", "wav",
            "--audio-quality", "0",       # Best quality
            "-o", str(output_path.with_suffix(".%(ext)s")),
            "--no-playlist",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            # Try with cookies if age-restricted
            print(f"  [WARN] Standard download failed, trying with cookies...")
            cmd.insert(1, "--cookies-from-browser")
            cmd.insert(2, "firefox")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        # yt-dlp adds extension, check for actual file
        for ext in ['.wav', '.mp3', '.m4a', '.webm']:
            candidate = output_path.with_suffix(ext)
            if candidate.exists():
                if ext != '.wav':
                    # Convert to WAV
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(candidate), 
                         "-ar", str(self.sample_rate), "-ac", "1", str(output_path)],
                        capture_output=True, timeout=120
                    )
                    candidate.unlink()
                break
    
    def _get_video_metadata(self, url: str) -> dict:
        """Get video metadata using yt-dlp."""
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-download", url],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            try:
                return json.loads(result.stdout.strip().split('\n')[0])
            except json.JSONDecodeError:
                pass
        return {}
    
    def _segment_speech(self, audio_path: Path, video_id: str, video_index: int) -> list:
        """Segment audio by voice activity using ffmpeg silenceremove + energy detection."""
        segments = []
        
        # Use ffmpeg to detect silence and split
        # First pass: remove silence > 0.5s, keep speech > 0.3s
        temp_dir = self.segments_dir / video_id
        temp_dir.mkdir(exist_ok=True)
        
        # Split on silence using ffmpeg segment filter
        segment_pattern = str(temp_dir / "seg_%04d.wav")
        
        # Detect silence points
        silence_result = subprocess.run(
            ["ffmpeg", "-i", str(audio_path),
             "-af", "silencedetect=noise=-30dB:d=0.5",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=300
        )
        
        # Parse silence detection output
        silence_starts = []
        silence_ends = []
        for line in silence_result.stderr.split('\n'):
            if 'silence_start:' in line:
                m = re.search(r'silence_start: ([\d.]+)', line)
                if m:
                    silence_starts.append(float(m.group(1)))
            elif 'silence_end:' in line:
                m = re.search(r'silence_end: ([\d.]+)', line)
                if m:
                    silence_ends.append(float(m.group(1)))
        
        # Get total duration
        duration_result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(audio_path)],
            capture_output=True, text=True, timeout=30
        )
        total_duration = float(duration_result.stdout.strip())
        
        # Invert silence to get speech segments
        speech_segments = []
        prev_end = 0.0
        for silence_start in silence_starts:
            if silence_start - prev_end > 0.3:  # Min 0.3s speech
                speech_segments.append((prev_end, silence_start))
            prev_end = silence_ends.pop(0) if silence_ends else silence_start
        # Last segment
        if total_duration - prev_end > 0.3:
            speech_segments.append((prev_end, total_duration))
        
        # Split audio into segments (max 15s each, split long segments)
        seg_idx = 0
        for start, end in speech_segments:
            duration = end - start
            if duration > 15.0:
                # Split long segments into 10s chunks
                for chunk_start in range(int(start), int(end) - 10, 8):
                    chunk_end = min(chunk_start + 15, int(end))
                    out_path = self.segments_dir / f"{video_id}_{video_index:03d}_{seg_idx:04d}.wav"
                    subprocess.run(
                        ["ffmpeg", "-y", "-ss", str(chunk_start), "-t", str(chunk_end - chunk_start),
                         "-i", str(audio_path), "-ar", str(self.sample_rate), "-ac", "1", str(out_path)],
                        capture_output=True, timeout=60
                    )
                    if out_path.exists():
                        segments.append({
                            "file": str(out_path),
                            "duration": chunk_end - chunk_start,
                            "source_video": video_id,
                        })
                        seg_idx += 1
            else:
                out_path = self.segments_dir / f"{video_id}_{video_index:03d}_{seg_idx:04d}.wav"
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
                     "-i", str(audio_path), "-ar", str(self.sample_rate), "-ac", "1", str(out_path)],
                    capture_output=True, timeout=60
                )
                if out_path.exists():
                    segments.append({
                        "file": str(out_path),
                        "duration": duration,
                        "source_video": video_id,
                    })
                    seg_idx += 1
        
        return segments
    
    def _quality_filter(self, segments: list, min_duration: float = 1.0, 
                        max_duration: float = 15.0) -> list:
        """Filter segments by quality criteria."""
        filtered = []
        for seg in segments:
            dur = seg["duration"]
            if dur < min_duration or dur > max_duration:
                # Remove too short/long segments
                Path(seg["file"]).unlink(missing_ok=True)
                continue
            
            # Check if file has reasonable energy (not just noise)
            result = subprocess.run(
                ["ffmpeg", "-i", str(seg["file"]),
                 "-af", "astats=metadata=1:reset=1", "-f", "null", "-"],
                capture_output=True, text=True, timeout=30
            )
            
            # Parse RMS level
            rms = -100  # Default low
            for line in result.stderr.split('\n'):
                if 'RMS level dB' in line:
                    try:
                        rms = float(line.split(':')[-1].strip())
                    except ValueError:
                        pass
            
            # Keep segments with reasonable RMS (not silence)
            if rms > -40:
                filtered.append(seg)
            else:
                Path(seg["file"]).unlink(missing_ok=True)
        
        return filtered
    
    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:v=)([a-zA-Z0-9_-]{11})',
            r'(?:youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'(?:embed/)([a-zA-Z0-9_-]{11})',
        ]
        for p in patterns:
            m = re.search(p, url)
            if m:
                return m.group(1)
        return "unknown"
    
    def create_csm_dataset(self, output_metadata: str = "csm_dataset.json"):
        """Create a CSM-1B compatible dataset from extracted segments."""
        segments = list(self.segments_dir.glob("*.wav"))
        
        dataset = []
        for seg in sorted(segments):
            # Create metadata entry for CSM fine-tuning
            # CSM uses interleaved text + audio, but for voice cloning
            # we mainly need the audio + speaker ID
            dataset.append({
                "path": str(seg),
                "speaker": "israel",  # Speaker ID for fine-tuning
                "duration": self._get_duration(seg),
            })
        
        output_path = self.output_dir / output_metadata
        with open(output_path, "w") as f:
            json.dump({
                "name": "israel_voice_csm",
                "sample_rate": self.sample_rate,
                "num_samples": len(dataset),
                "total_duration": sum(d["duration"] for d in dataset),
                "samples": dataset,
            }, f, indent=2)
        
        print(f"[YouTube Voice] CSM dataset created: {output_path}")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Total duration: {sum(d['duration'] for d in dataset):.1f}s")
        
        return output_path
    
    def _get_duration(self, audio_path: Path) -> float:
        """Get audio duration using ffprobe."""
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(audio_path)],
            capture_output=True, text=True, timeout=15
        )
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0


def main():
    parser = argparse.ArgumentParser(description="Extract voice samples from YouTube")
    parser.add_argument("url", help="YouTube video or playlist URL")
    parser.add_argument("--output-dir", default="./voice_data", help="Output directory")
    parser.add_argument("--max-videos", type=int, default=None, help="Max videos from playlist")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    args = parser.parse_args()
    
    extractor = YouTubeVoiceExtractor(args.output_dir, args.sample_rate)
    
    # Determine if playlist or single video
    if "list=" in args.url:
        segments = extractor.extract_from_playlist(args.url, max_videos=args.max_videos)
    else:
        segments = extractor.extract_from_video(args.url)
    
    if segments:
        extractor.create_csm_dataset()
        print(f"\n✅ Voice extraction complete. {len(segments)} segments ready for fine-tuning.")
    else:
        print("\n❌ No segments extracted. Check the URL and try again.")


if __name__ == "__main__":
    main()
