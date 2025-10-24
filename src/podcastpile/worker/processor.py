#!/usr/bin/env python3
"""
Worker processor for Podcast Pile - diarizes audio and uploads results
"""

import os
import json
import hashlib
import tempfile
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import requests
import librosa
import soundfile as sf
from nemo.collections.asr.models import SortformerEncLabelModel
import nemo.collections.asr as nemo_asr

logger = logging.getLogger(__name__)


def get_gpu_info() -> Optional[str]:
    """Get GPU device information"""
    try:
        import torch
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device_id)
            return f"{gpu_name} (CUDA {device_id})"
    except Exception as e:
        logger.debug(f"Could not get GPU info: {e}")
    return None


class AudioProcessor:
    """Handles audio processing and diarization"""

    def __init__(self, config: str = "high_latency", model_path: Optional[str] = None):
        """
        Initialize audio processor with models

        Args:
            config: Streaming configuration (very_high_latency, high_latency, low_latency, ultra_low_latency)
            model_path: Optional path to custom .nemo model file
        """
        self.config_name = config
        self.configs = {
            "very_high_latency": {
                "CHUNK_SIZE": 340, "RIGHT_CONTEXT": 40, "FIFO_SIZE": 40,
                "UPDATE_PERIOD": 300, "SPEAKER_CACHE_SIZE": 188
            },
            "high_latency": {
                "CHUNK_SIZE": 124, "RIGHT_CONTEXT": 1, "FIFO_SIZE": 124,
                "UPDATE_PERIOD": 124, "SPEAKER_CACHE_SIZE": 188
            },
            "low_latency": {
                "CHUNK_SIZE": 6, "RIGHT_CONTEXT": 7, "FIFO_SIZE": 188,
                "UPDATE_PERIOD": 144, "SPEAKER_CACHE_SIZE": 188
            },
            "ultra_low_latency": {
                "CHUNK_SIZE": 3, "RIGHT_CONTEXT": 1, "FIFO_SIZE": 188,
                "UPDATE_PERIOD": 144, "SPEAKER_CACHE_SIZE": 188
            }
        }

        self.diar_model = None
        self.asr_model = None
        self.model_path = model_path

    def load_models(self):
        """Load diarization and ASR models"""
        logger.info("Loading diarization model...")
        if self.model_path and os.path.exists(self.model_path):
            self.diar_model = SortformerEncLabelModel.restore_from(
                restore_path=self.model_path,
                map_location='cuda',
                strict=False
            )
        else:
            self.diar_model = SortformerEncLabelModel.from_pretrained(
                "nvidia/diar_streaming_sortformer_4spk-v2"
            )

        self.diar_model.eval()
        logger.info("✓ Diarization model loaded")

        logger.info("Loading ASR model...")
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v3"
        )
        self.asr_model.eval()
        logger.info("✓ ASR model loaded")

        # Set streaming configuration
        config = self.configs[self.config_name]
        self.diar_model.sortformer_modules.chunk_len = config["CHUNK_SIZE"]
        self.diar_model.sortformer_modules.chunk_right_context = config["RIGHT_CONTEXT"]
        self.diar_model.sortformer_modules.fifo_len = config["FIFO_SIZE"]
        self.diar_model.sortformer_modules.spkcache_update_period = config["UPDATE_PERIOD"]
        self.diar_model.sortformer_modules.spkcache_len = config["SPEAKER_CACHE_SIZE"]
        self.diar_model.sortformer_modules._check_streaming_parameters()
        logger.info(f"✓ Using {self.config_name} configuration")

    @staticmethod
    def convert_to_mono_if_needed(audio_path: str) -> str:
        """Convert stereo to mono if needed"""
        audio, sr = librosa.load(audio_path, sr=None, mono=False)

        if audio.ndim > 1:
            logger.info(f"Converting {audio_path} to mono...")
            mono_path = str(Path(audio_path).with_suffix('')) + '_mono.wav'
            audio_mono = librosa.to_mono(audio)
            sf.write(mono_path, audio_mono, sr)
            return mono_path

        return audio_path

    @staticmethod
    def extract_audio_segment(audio, sr: int, start_time: float, end_time: float):
        """Extract a segment from audio array based on timestamps"""
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        return audio[start_sample:end_sample]

    @staticmethod
    def compute_file_hashes(filepath: str) -> Dict[str, str]:
        """Compute SHA256 and MD5 hashes of a file"""
        sha256_hash = hashlib.sha256()
        md5_hash = hashlib.md5()

        with open(filepath, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
                md5_hash.update(byte_block)

        return {
            "sha256": sha256_hash.hexdigest(),
            "md5": md5_hash.hexdigest()
        }

    def diarize_audio(self, audio_path: str, episode_url: str = None, language: str = None) -> Dict:
        """
        Diarize a single audio file and return results

        Args:
            audio_path: Path to audio file
            episode_url: Original episode URL
            language: Language code

        Returns:
            Dict with diarization results, transcriptions, file hashes, and metadata
        """
        start_time = time.time()
        # Compute hashes first (before any conversion)
        logger.info(f"Computing file hashes for {audio_path}...")
        hashes = self.compute_file_hashes(audio_path)
        logger.info(f"SHA256: {hashes['sha256']}")
        logger.info(f"MD5: {hashes['md5']}")

        # Convert to mono if needed
        audio_path = self.convert_to_mono_if_needed(audio_path)

        # Get audio duration
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(audio) / sr

        logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

        # Perform diarization
        logger.info("Diarizing...")
        segments = self.diar_model.diarize(audio=audio_path, batch_size=1)

        # Parse segments into structured format
        results = []
        segment_list = segments[0] if isinstance(segments, list) and len(segments) > 0 else segments

        for seg in segment_list:
            if isinstance(seg, str):
                parts = seg.split()
                start = float(parts[0])
                end = float(parts[1])
                speaker = parts[2].replace('speaker_', '')  # Remove 'speaker_' prefix
                results.append({
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "duration": end - start
                })

        # Extract audio segments and save to temp files
        logger.info(f"Extracting {len(results)} segments...")
        temp_files = []
        for i, segment in enumerate(results):
            segment_audio = self.extract_audio_segment(audio, sr, segment["start"], segment["end"])
            temp_path = f"/tmp/segment_{i}_{os.getpid()}.wav"
            sf.write(temp_path, segment_audio, 16000)
            temp_files.append(temp_path)

        try:
            # Transcribe all segments at once
            logger.info(f"Transcribing {len(temp_files)} segments...")
            batch_results = self.asr_model.transcribe(temp_files)
            transcriptions = [result.text for result in batch_results]

            # Add transcriptions to results
            for i in range(len(results)):
                results[i]["transcription"] = transcriptions[i]

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Get GPU info
        gpu_info = get_gpu_info()

        # Create output record with all metadata
        output_record = {
            "audio_filepath": str(Path(audio_path).absolute()),
            "episode_url": episode_url,
            "language": language,
            "duration": duration,
            "num_segments": len(results),
            "segments": results,
            "file_hashes": hashes,
            "num_speakers": len(set(s['speaker'] for s in results)),
            "processing_time": processing_time,
            "gpu_info": gpu_info,
            "processed_at": datetime.utcnow().isoformat() + "Z"
        }

        logger.info(f"✓ Processed {len(results)} segments in {processing_time:.2f}s")
        logger.info(f"  Speakers detected: {output_record['num_speakers']}")
        if gpu_info:
            logger.info(f"  GPU: {gpu_info}")

        return output_record


class PodcastPileWorker:
    """Worker that fetches jobs and processes them"""

    def __init__(
        self,
        manager_url: str,
        worker_id: str,
        worker_password: Optional[str] = None,
        config: str = "high_latency",
        model_path: Optional[str] = None
    ):
        """
        Initialize worker

        Args:
            manager_url: URL of the manager server
            worker_id: Unique identifier for this worker
            worker_password: Optional password for worker authentication
            config: Diarization config (very_high_latency, high_latency, low_latency, ultra_low_latency)
            model_path: Optional path to custom .nemo model file
        """
        self.manager_url = manager_url.rstrip('/')
        self.worker_id = worker_id
        self.worker_password = worker_password
        self.processor = AudioProcessor(config=config, model_path=model_path)

        # Headers for API requests
        self.headers = {}
        if worker_password:
            self.headers["X-Worker-Password"] = worker_password

    def load_models(self):
        """Load processing models"""
        self.processor.load_models()

    def request_job(self, languages: str = "en") -> Optional[Dict]:
        """
        Request a job from the manager

        Args:
            languages: Comma-separated language codes (default: "en" for English only)

        Returns:
            Job dict or None if no jobs available
        """
        url = f"{self.manager_url}/api/jobs/request"
        params = {
            "worker_id": self.worker_id,
            "languages": languages
        }

        try:
            response = requests.post(url, params=params, headers=self.headers)

            if response.status_code == 404:
                logger.info("No jobs available")
                return None

            response.raise_for_status()
            job = response.json()
            logger.info(f"Received job #{job['job_id']}: {job['episode_url']}")
            return job

        except requests.exceptions.RequestException as e:
            logger.error(f"Error requesting job: {e}")
            return None

    def download_audio(self, url: str, temp_dir: str) -> str:
        """
        Download audio file from URL

        Args:
            url: Audio file URL
            temp_dir: Directory to save file

        Returns:
            Path to downloaded file
        """
        logger.info(f"Downloading audio from {url}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Determine file extension from URL or content-type
        ext = ".mp3"  # Default
        if url.endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
            ext = Path(url).suffix

        filepath = os.path.join(temp_dir, f"audio{ext}")

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"✓ Downloaded to {filepath}")
        return filepath

    def update_job_status(self, job_id: int, status: str) -> bool:
        """Update job status on manager"""
        url = f"{self.manager_url}/api/jobs/{job_id}/status"

        try:
            response = requests.post(
                url,
                json={"status": status},
                headers=self.headers
            )
            response.raise_for_status()
            logger.info(f"Updated job #{job_id} status to {status}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Error updating job status: {e}")
            return False

    def submit_results(self, job_id: int, results: Dict) -> bool:
        """
        Submit processing results to manager

        Args:
            job_id: Job ID
            results: Processing results dict

        Returns:
            True if successful
        """
        url = f"{self.manager_url}/api/jobs/{job_id}/complete"

        # Format the results for submission
        payload = {
            "result_json": json.dumps(results),
            "transcription": self._format_transcription(results),
            "diarization": self._format_diarization(results),
            "processing_duration": results.get("processing_time"),
            "worker_gpu": results.get("gpu_info"),
            "processed_at": results.get("processed_at")
        }

        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            logger.info(f"✓ Submitted results for job #{job_id}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Error submitting results: {e}")
            return False

    def report_failure(self, job_id: int, error_message: str) -> bool:
        """Report job failure to manager"""
        url = f"{self.manager_url}/api/jobs/{job_id}/fail"

        try:
            response = requests.post(
                url,
                json={"error": error_message},
                headers=self.headers
            )
            response.raise_for_status()
            logger.info(f"Reported failure for job #{job_id}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Error reporting failure: {e}")
            return False

    @staticmethod
    def _format_transcription(results: Dict) -> str:
        """Format transcription as plain text"""
        lines = []
        for segment in results.get("segments", []):
            speaker = segment["speaker"]
            text = segment["transcription"]
            lines.append(f"Speaker {speaker}: {text}")
        return "\n".join(lines)

    @staticmethod
    def _format_diarization(results: Dict) -> str:
        """Format diarization timestamps as plain text"""
        lines = []
        for segment in results.get("segments", []):
            start = segment["start"]
            end = segment["end"]
            speaker = segment["speaker"]
            lines.append(f"{start:.2f} {end:.2f} speaker_{speaker}")
        return "\n".join(lines)

    def process_job(self, job: Dict) -> bool:
        """
        Process a single job

        Args:
            job: Job dict from manager

        Returns:
            True if successful
        """
        job_id = job["job_id"]
        episode_url = job["episode_url"]
        language = job.get("language")

        logger.info(f"Processing job #{job_id}...")

        # Update status to processing
        self.update_job_status(job_id, "processing")

        temp_dir = tempfile.mkdtemp()

        try:
            # Download audio
            audio_path = self.download_audio(episode_url, temp_dir)

            # Process audio with metadata
            results = self.processor.diarize_audio(
                audio_path,
                episode_url=episode_url,
                language=language
            )

            # Submit results
            success = self.submit_results(job_id, results)

            return success

        except Exception as e:
            logger.error(f"Error processing job #{job_id}: {e}", exc_info=True)
            self.report_failure(job_id, str(e))
            return False

        finally:
            # Cleanup temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def run_once(self, languages: str = "en") -> bool:
        """
        Request and process a single job

        Args:
            languages: Comma-separated language codes (default: "en")

        Returns:
            True if job was processed, False if no job available
        """
        job = self.request_job(languages=languages)

        if not job:
            return False

        self.process_job(job)
        return True

    def run_loop(self, languages: str = "en", poll_interval: int = 10):
        """
        Continuously request and process jobs

        Args:
            languages: Comma-separated language codes (default: "en")
            poll_interval: Seconds to wait between requests if no job available
        """
        import time

        logger.info(f"Starting worker loop (languages: {languages})...")
        logger.info("Press Ctrl+C to stop")

        try:
            while True:
                job = self.request_job(languages=languages)

                if job:
                    self.process_job(job)
                else:
                    logger.info(f"No jobs available, waiting {poll_interval}s...")
                    time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("\nStopping worker...")
