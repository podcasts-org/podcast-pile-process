#!/usr/bin/env python3
"""
Worker processor for Podcast Pile - diarizes audio and uploads results
"""

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import librosa
import nemo.collections.asr as nemo_asr
import numpy as np
import requests
import soundfile as sf
import torch
from nemo.collections.asr.models import SortformerEncLabelModel
from tqdm import tqdm
from podcastpile.nisqa import NISQAPredictor

logger = logging.getLogger(__name__)

# Worker version - increment when making changes to processing logic
WORKER_VERSION = "0.1.4"  # Added clipping detection and loudness metrics


def get_gpu_info(gpu_id: Optional[int] = None) -> Optional[str]:
    """Get GPU device information"""
    try:
        import torch

        if torch.cuda.is_available():
            device_id = gpu_id if gpu_id is not None else torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device_id)
            return f"{gpu_name} (CUDA {device_id})"
    except Exception as e:
        logger.debug(f"Could not get GPU info: {e}")
    return None


def get_available_gpus() -> list:
    """Get list of available GPU IDs"""
    try:
        import torch

        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except Exception:
        pass
    return []


class AudioProcessor:
    """Handles audio processing and diarization"""

    def __init__(
        self,
        config: str = "high_latency",
        model_path: Optional[str] = None,
        gpu_id: Optional[int] = None,
        languages: str = "en",
        batch_size: int = 4,
    ):
        """
        Initialize audio processor with models

        Args:
            config: Streaming configuration (very_high_latency, high_latency, low_latency, ultra_low_latency)
            model_path: Optional path to custom .nemo model file
            gpu_id: GPU device ID to use (None for auto-select)
            languages: Comma-separated language codes to determine which models to load
            batch_size: Batch size for FireRedASR transcription (1, 2, 4, 8, 16, etc.) Default: 4
        """
        self.config_name = config
        self.gpu_id = gpu_id
        self.languages = [lang.strip().lower() for lang in languages.split(",")]
        self.batch_size = batch_size
        self.configs = {
            "very_high_latency": {
                "CHUNK_SIZE": 340,
                "RIGHT_CONTEXT": 40,
                "FIFO_SIZE": 40,
                "UPDATE_PERIOD": 300,
                "SPEAKER_CACHE_SIZE": 188,
            },
            "high_latency": {
                "CHUNK_SIZE": 124,
                "RIGHT_CONTEXT": 1,
                "FIFO_SIZE": 124,
                "UPDATE_PERIOD": 124,
                "SPEAKER_CACHE_SIZE": 188,
            },
            "low_latency": {
                "CHUNK_SIZE": 6,
                "RIGHT_CONTEXT": 7,
                "FIFO_SIZE": 188,
                "UPDATE_PERIOD": 144,
                "SPEAKER_CACHE_SIZE": 188,
            },
            "ultra_low_latency": {
                "CHUNK_SIZE": 3,
                "RIGHT_CONTEXT": 1,
                "FIFO_SIZE": 188,
                "UPDATE_PERIOD": 144,
                "SPEAKER_CACHE_SIZE": 188,
            },
        }

        self.diar_model = None
        self.asr_model = None  # Parakeet (English)
        self.zh_asr_model = None  # Paraformer (Chinese)
        self.bgm_classifier = None  # BGM detection
        self.bgm_model_id = "podcasts-org/detect-background-music"  # HF model ID
        self.nisqa_predictor = None  # NISQA quality assessment
        self.model_path = model_path

    def load_models(self):
        """Load diarization and ASR models"""
        # Set GPU device if specified
        if self.gpu_id is not None:
            try:
                import torch

                torch.cuda.set_device(self.gpu_id)
                logger.info(
                    f"Using GPU {self.gpu_id}: {torch.cuda.get_device_name(self.gpu_id)}"
                )
            except Exception as e:
                logger.warning(f"Could not set GPU {self.gpu_id}: {e}")

        logger.info("Loading diarization model...")

        # Determine map location for model loading
        if self.gpu_id is not None:
            map_location = f"cuda:{self.gpu_id}"
        else:
            map_location = "cpu"

        if self.model_path and os.path.exists(self.model_path):
            self.diar_model = SortformerEncLabelModel.restore_from(
                restore_path=self.model_path, map_location=map_location, strict=False
            )
        else:
            self.diar_model = SortformerEncLabelModel.from_pretrained(
                "nvidia/diar_streaming_sortformer_4spk-v2"
            )
            # Move to correct GPU if specified
            if self.gpu_id is not None:
                self.diar_model = self.diar_model.to(map_location)

        self.diar_model.eval()
        logger.info(f"✓ Diarization model loaded on {map_location}")

        # Determine which ASR models to load based on languages
        needs_chinese = "zh" in self.languages or "cn" in self.languages
        needs_english = any(lang not in ["zh", "cn"] for lang in self.languages)

        # Load Parakeet (English) if needed
        if needs_english:
            logger.info("Loading Parakeet ASR model (English)...")
            self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/parakeet-tdt-0.6b-v3"
            )
            # Move to correct GPU if specified
            if self.gpu_id is not None:
                self.asr_model = self.asr_model.to(f"cuda:{self.gpu_id}")
            self.asr_model.eval()
            logger.info(f"✓ Parakeet ASR model loaded on {map_location}")

        # Load Paraformer (Chinese) if needed
        if needs_chinese:
            logger.info("Loading Paraformer model (Chinese)...")
            try:
                from funasr import AutoModel

                self.zh_asr_model = AutoModel(
                    model="paraformer-zh",
                    model_revision="v2.0.4",
                    vad_model="fsmn-vad",
                    vad_model_revision="v2.0.4",
                    punc_model="ct-punc-c",
                    punc_model_revision="v2.0.4",
                    hub="hf",
                    device=f"cuda:{self.gpu_id}" if self.gpu_id is not None else "cpu",
                )
                logger.info(f"✓ Paraformer model loaded on {map_location}")
            except ImportError as e:
                logger.error(f"Failed to import FunASR: {e}")
                logger.error("Install with: pip install funasr")
                raise
            except Exception as e:
                logger.error(f"Failed to load Paraformer model: {e}")
                raise

        # Load BGM classifier
        logger.info("Loading BGM classifier...")
        try:
            from transformers import pipeline

            self.bgm_classifier = pipeline(
                "audio-classification",
                model=self.bgm_model_id,
                device=self.gpu_id if self.gpu_id is not None else -1,
            )
            logger.info(f"✓ BGM classifier loaded ({self.bgm_model_id})")
        except Exception as e:
            logger.error(f"Failed to load BGM classifier: {e}")
            raise

        # Load NISQA quality assessment model
        logger.info("Loading NISQA quality assessment model...")
        try:
            device = torch.device(f"cuda:{self.gpu_id}" if self.gpu_id is not None else "cuda" if torch.cuda.is_available() else "cpu")
            self.nisqa_predictor = NISQAPredictor(device=device, dim=True)
            logger.info(f"✓ NISQA model loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load NISQA model: {e}")
            raise

        # Set streaming configuration
        config = self.configs[self.config_name]
        self.diar_model.sortformer_modules.chunk_len = config["CHUNK_SIZE"]
        self.diar_model.sortformer_modules.chunk_right_context = config["RIGHT_CONTEXT"]
        self.diar_model.sortformer_modules.fifo_len = config["FIFO_SIZE"]
        self.diar_model.sortformer_modules.spkcache_update_period = config[
            "UPDATE_PERIOD"
        ]
        self.diar_model.sortformer_modules.spkcache_len = config["SPEAKER_CACHE_SIZE"]
        self.diar_model.sortformer_modules._check_streaming_parameters()
        logger.info(f"✓ Using {self.config_name} configuration")

    @staticmethod
    def convert_to_mono_if_needed(audio_path: str) -> str:
        """Convert stereo to mono if needed"""
        audio, sr = librosa.load(audio_path, sr=None, mono=False)

        if audio.ndim > 1:
            logger.info(f"Converting {audio_path} to mono...")
            mono_path = str(Path(audio_path).with_suffix("")) + "_mono.wav"
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

        return {"sha256": sha256_hash.hexdigest(), "md5": md5_hash.hexdigest()}

    def diarize_audio(
        self, audio_path: str, episode_url: str = None, language: str = None,
        converted_audio_path: str = None
    ) -> Dict:
        """
        Diarize a single audio file and return results

        Args:
            audio_path: Path to original audio file (used for hashing)
            episode_url: Original episode URL
            language: Language code
            converted_audio_path: Optional pre-converted mono audio path (from prefetch)

        Returns:
            Dict with diarization results, transcriptions, file hashes, and metadata
        """
        start_time = time.time()
        # Compute hashes on the ORIGINAL audio file (before any conversion)
        logger.info(f"Computing file hashes for {audio_path}...")
        hashes = self.compute_file_hashes(audio_path)
        logger.info(f"SHA256: {hashes['sha256']}")
        logger.info(f"MD5: {hashes['md5']}")

        # Use pre-converted path if provided, otherwise convert now
        if converted_audio_path:
            logger.info(f"Using pre-converted audio: {converted_audio_path}")
            processing_audio_path = converted_audio_path
        else:
            processing_audio_path = self.convert_to_mono_if_needed(audio_path)

        # Get audio duration
        audio, sr = librosa.load(processing_audio_path, sr=16000, mono=True)
        duration = len(audio) / sr

        logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

        # Perform diarization
        logger.info("Diarizing...")
        segments = self.diar_model.diarize(audio=processing_audio_path, batch_size=1)

        # Parse segments into structured format
        results = []
        segment_list = (
            segments[0]
            if isinstance(segments, list) and len(segments) > 0
            else segments
        )

        for seg in segment_list:
            if isinstance(seg, str):
                parts = seg.split()
                start = float(parts[0])
                end = float(parts[1])
                speaker = parts[2].replace("speaker_", "")  # Remove 'speaker_' prefix
                results.append(
                    {
                        "start": start,
                        "end": end,
                        "speaker": speaker,
                        "duration": end - start,
                    }
                )

        # Extract audio segments and save to temp files
        logger.info(f"Extracting {len(results)} segments...")
        temp_files = []
        session_id = uuid.uuid4().hex[:8]  # Unique ID for this diarization session
        for i, segment in enumerate(results):
            segment_audio = self.extract_audio_segment(
                audio, sr, segment["start"], segment["end"]
            )

            # Compute clipping and loudness metrics for this segment
            # Clipping detection: count samples at or near maximum amplitude
            clipped_samples = np.sum(np.abs(segment_audio) >= 0.99)
            clip_rate = float(clipped_samples / len(segment_audio)) if len(segment_audio) > 0 else 0.0

            # Loudness metrics
            rms = np.sqrt(np.mean(segment_audio**2))
            rms_db = 20 * np.log10(rms) if rms > 0 else -100.0
            peak = np.max(np.abs(segment_audio))
            peak_db = 20 * np.log10(peak) if peak > 0 else -100.0

            # Add to segment metadata
            segment["clip_rate"] = clip_rate
            segment["clipped_samples"] = int(clipped_samples)
            segment["has_clipping"] = clip_rate > 0.001  # >0.1% clipping threshold
            segment["rms_db"] = float(rms_db)
            segment["peak_db"] = float(peak_db)
            segment["dynamic_range_db"] = float(peak_db - rms_db)

            temp_path = f"/tmp/segment_{session_id}_{i}.wav"
            sf.write(temp_path, segment_audio, 16000)
            temp_files.append(temp_path)

        try:
            # Transcribe all segments at once using appropriate model
            is_chinese = language and (
                language.lower() == "zh" or language.lower() == "cn"
            )

            logger.info(f"Transcribing {len(temp_files)} segments...")
            if is_chinese and self.zh_asr_model:
                # Use Paraformer for Chinese - process all files at once
                logger.info("Using Paraformer for Chinese transcription")

                # Pass all temp files at once
                paraformer_results = self.zh_asr_model.generate(
                    input=temp_files, batch_size_s=300
                )

                # Extract text from results - format is [{'key': 'filename', 'text': 'transcription'}, ...]
                transcriptions = []
                for result in paraformer_results:
                    if isinstance(result, dict):
                        transcriptions.append(result.get("text", ""))
                    else:
                        transcriptions.append("")

            elif self.asr_model:
                # Use Parakeet for English/other languages
                logger.info("Using Parakeet for transcription")
                batch_results = self.asr_model.transcribe(temp_files)
                transcriptions = [result.text for result in batch_results]
            else:
                logger.error("No ASR model available for this language")
                transcriptions = ["" for _ in temp_files]

            # Add transcriptions to results
            logger.info(
                f"Adding {len(transcriptions)} transcriptions to {len(results)} segments"
            )
            for i in range(len(results)):
                results[i]["transcription"] = transcriptions[i]
                if i < 3:  # Log first 3 for debugging
                    logger.debug(
                        f"Segment {i}: '{transcriptions[i][:50] if transcriptions[i] else '(empty)'}'..."
                    )

            # BGM classification for all segments (BATCHED for GPU efficiency)
            logger.info(f"Classifying BGM for {len(temp_files)} segments...")

            # Load all segment audio files
            audio_inputs = []
            for temp_file in temp_files:
                try:
                    segment_audio, segment_sr = librosa.load(
                        temp_file, sr=16000, mono=True
                    )

                    # Normalize to float32
                    if segment_audio.dtype == np.int16:
                        segment_audio = segment_audio.astype(np.float32) / 32768.0
                    elif segment_audio.dtype == np.int32:
                        segment_audio = segment_audio.astype(np.float32) / 2147483648.0

                    audio_inputs.append(
                        {"array": segment_audio, "sampling_rate": segment_sr}
                    )
                except Exception as e:
                    logger.warning(f"Failed to load audio for BGM classification: {e}")
                    audio_inputs.append(None)

            # Process in batches for GPU efficiency
            try:
                # Filter out None values and track indices
                valid_inputs = []
                valid_indices = []
                for i, audio_input in enumerate(audio_inputs):
                    if audio_input is not None:
                        valid_inputs.append(audio_input)
                        valid_indices.append(i)
                    else:
                        results[i]["bgm_probability"] = 0.0
                        results[i]["bgm"] = False

                # Batch classify all valid inputs at once
                if valid_inputs:
                    batch_predictions = self.bgm_classifier(valid_inputs, batch_size=32)

                    # Map predictions back to results
                    for idx, predictions in zip(valid_indices, batch_predictions):
                        bgm_prob = 0.0
                        for pred in predictions:
                            if pred["label"] == "bgm":
                                bgm_prob = pred["score"]
                                break
                        results[idx]["bgm_probability"] = bgm_prob
                        results[idx]["bgm"] = bgm_prob > 0.5

                logger.info(f"✓ Classified {len(valid_inputs)} segments")
            except Exception as e:
                logger.warning(f"Batch BGM classification failed: {e}, falling back to defaults")
                for i in range(len(results)):
                    if "bgm_probability" not in results[i]:
                        results[i]["bgm_probability"] = 0.0
                        results[i]["bgm"] = False

            # NISQA quality assessment for all segments (BATCHED for GPU efficiency)
            logger.info(f"Assessing audio quality (NISQA) for {len(temp_files)} segments...")

            try:
                # Prepare arrays for batch prediction
                valid_arrays = []
                valid_indices = []

                for i, audio_input in enumerate(audio_inputs):
                    if audio_input is not None:
                        valid_arrays.append(audio_input["array"])
                        valid_indices.append(i)
                    else:
                        # Set defaults for failed segments
                        results[i]["quality_mos"] = None
                        results[i]["quality_noisiness"] = None
                        results[i]["quality_discontinuity"] = None
                        results[i]["quality_coloration"] = None
                        results[i]["quality_loudness"] = None

                if valid_arrays:
                    # Batch predict using official NISQA implementation
                    # Use larger batch size for better GPU utilization
                    predictions = self.nisqa_predictor.predict_arrays(
                        audio_arrays=valid_arrays,
                        sample_rate=16000,
                        batch_size=16
                    )

                    # Map predictions back to results
                    successful_assessments = 0
                    for idx, pred_idx in enumerate(valid_indices):
                        try:
                            results[pred_idx]["quality_mos"] = float(predictions['mos'][idx])
                            results[pred_idx]["quality_noisiness"] = float(predictions['noisiness'][idx])
                            results[pred_idx]["quality_discontinuity"] = float(predictions['discontinuity'][idx])
                            results[pred_idx]["quality_coloration"] = float(predictions['coloration'][idx])
                            results[pred_idx]["quality_loudness"] = float(predictions['loudness'][idx])
                            successful_assessments += 1
                        except Exception as e:
                            logger.warning(f"Failed to process NISQA result for segment {pred_idx}: {e}")
                            results[pred_idx]["quality_mos"] = None
                            results[pred_idx]["quality_noisiness"] = None
                            results[pred_idx]["quality_discontinuity"] = None
                            results[pred_idx]["quality_coloration"] = None
                            results[pred_idx]["quality_loudness"] = None

                    logger.info(f"✓ Assessed quality for {successful_assessments}/{len(valid_arrays)} segments")
                else:
                    logger.info("No valid segments for NISQA assessment")

            except Exception as e:
                logger.warning(f"Batch NISQA assessment failed: {e}, setting defaults")
                import traceback
                traceback.print_exc()
                for i in range(len(results)):
                    if "quality_mos" not in results[i]:
                        results[i]["quality_mos"] = None
                        results[i]["quality_noisiness"] = None
                        results[i]["quality_discontinuity"] = None
                        results[i]["quality_coloration"] = None
                        results[i]["quality_loudness"] = None

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Get GPU info
        gpu_info = get_gpu_info(self.gpu_id)

        # Compute episode-level quality statistics from segment scores
        quality_mos_scores = [s.get("quality_mos") for s in results if s.get("quality_mos") is not None]
        quality_noisiness_scores = [s.get("quality_noisiness") for s in results if s.get("quality_noisiness") is not None]
        quality_discontinuity_scores = [s.get("quality_discontinuity") for s in results if s.get("quality_discontinuity") is not None]
        quality_coloration_scores = [s.get("quality_coloration") for s in results if s.get("quality_coloration") is not None]
        quality_loudness_scores = [s.get("quality_loudness") for s in results if s.get("quality_loudness") is not None]

        # Clipping and loudness aggregations (per-segment values)
        clip_rates = [s.get("clip_rate", 0.0) for s in results]
        rms_db_values = [s.get("rms_db") for s in results if s.get("rms_db") is not None and s.get("rms_db") != -100.0]
        peak_db_values = [s.get("peak_db") for s in results if s.get("peak_db") is not None and s.get("peak_db") != -100.0]
        dynamic_range_values = [s.get("dynamic_range_db") for s in results if s.get("dynamic_range_db") is not None]

        episode_quality = {}
        if quality_mos_scores:
            episode_quality = {
                # NISQA scores
                "mean_mos": float(np.mean(quality_mos_scores)),
                "median_mos": float(np.median(quality_mos_scores)),
                "p25_mos": float(np.percentile(quality_mos_scores, 25)),
                "p75_mos": float(np.percentile(quality_mos_scores, 75)),
                "min_mos": float(np.min(quality_mos_scores)),
                "max_mos": float(np.max(quality_mos_scores)),
                "mean_noisiness": float(np.mean(quality_noisiness_scores)),
                "mean_discontinuity": float(np.mean(quality_discontinuity_scores)),
                "mean_coloration": float(np.mean(quality_coloration_scores)),
                "mean_loudness": float(np.mean(quality_loudness_scores)),
                # Quality tier counts
                "high_quality_segments": sum(1 for s in quality_mos_scores if s >= 4.0),
                "medium_quality_segments": sum(1 for s in quality_mos_scores if 3.0 <= s < 4.0),
                "low_quality_segments": sum(1 for s in quality_mos_scores if s < 3.0),
            }

        # Clipping statistics
        clipping_stats = {
            "mean_clip_rate": float(np.mean(clip_rates)) if clip_rates else 0.0,
            "max_clip_rate": float(np.max(clip_rates)) if clip_rates else 0.0,
            "segments_with_clipping": sum(1 for s in results if s.get("has_clipping", False)),
            "total_clipped_samples": sum(s.get("clipped_samples", 0) for s in results),
        }

        # Loudness statistics
        loudness_stats = {}
        if rms_db_values:
            loudness_stats = {
                "mean_rms_db": float(np.mean(rms_db_values)),
                "median_rms_db": float(np.median(rms_db_values)),
                "min_rms_db": float(np.min(rms_db_values)),
                "max_rms_db": float(np.max(rms_db_values)),
                "mean_peak_db": float(np.mean(peak_db_values)),
                "max_peak_db": float(np.max(peak_db_values)),
                "mean_dynamic_range_db": float(np.mean(dynamic_range_values)),
                "rms_variation": float(np.std(rms_db_values)),  # Loudness consistency
            }

        # Create output record with all metadata
        output_record = {
            "audio_filepath": str(Path(audio_path).absolute()),
            "episode_url": episode_url,
            "language": language,
            "duration": duration,
            "num_segments": len(results),
            "segments": results,
            "file_hashes": hashes,
            "num_speakers": len(set(s["speaker"] for s in results)),
            "processing_time": processing_time,
            "gpu_info": gpu_info,
            "bgm_model": self.bgm_model_id,
            "worker_version": WORKER_VERSION,
            "episode_quality": episode_quality,
            "clipping_stats": clipping_stats,
            "loudness_stats": loudness_stats,
            "processed_at": datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z'),
        }

        logger.info(f"✓ Processed {len(results)} segments in {processing_time:.2f}s")
        logger.info(f"  Speakers detected: {output_record['num_speakers']}")
        if episode_quality:
            logger.info(f"  Quality: MOS={episode_quality['mean_mos']:.2f} (median={episode_quality['median_mos']:.2f})")
        if clipping_stats["segments_with_clipping"] > 0:
            logger.info(f"  Clipping: {clipping_stats['segments_with_clipping']}/{len(results)} segments, max rate={clipping_stats['max_clip_rate']:.4f}")
        if loudness_stats:
            logger.info(f"  Loudness: mean={loudness_stats['mean_rms_db']:.1f}dB, peak={loudness_stats['max_peak_db']:.1f}dB")
        if gpu_info:
            logger.info(f"  GPU: {gpu_info}")

        return output_record


class S3Uploader:
    """Handles S3 uploads in a background thread"""

    def __init__(self, s3_config: Dict):
        """
        Initialize S3 uploader

        Args:
            s3_config: Dict with keys: endpoint_url, access_key_id, secret_access_key, bucket, region
        """
        import boto3
        from botocore.client import Config

        self.bucket = s3_config["bucket"]
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=s3_config["endpoint_url"],
            aws_access_key_id=s3_config["access_key_id"],
            aws_secret_access_key=s3_config["secret_access_key"],
            config=Config(signature_version="s3v4"),
            region_name=s3_config["region"],
        )
        logger.info(f"S3 uploader initialized (bucket: {self.bucket})")

    def compress_audio(self, input_path: str) -> str:
        """
        Compress audio to MP3 with fast, reasonable settings for podcasts

        Args:
            input_path: Path to input audio file

        Returns:
            Path to compressed MP3 file
        """
        import subprocess

        # Create compressed filename with UUID to avoid conflicts
        base_path = Path(input_path).with_suffix("")
        unique_id = uuid.uuid4().hex[:8]  # Use first 8 chars of UUID for brevity
        compressed_path = f"{base_path}_compressed_{unique_id}.mp3"

        # Use ffmpeg with fast preset, variable bitrate optimized for voice
        # -q:a 4 gives VBR ~128kbps average (good for podcasts, not music)
        # -preset fast prioritizes speed over compression efficiency
        # -ac 1 converts to mono (podcasts rarely need stereo)
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-q:a", "4",  # VBR quality (~128kbps, good for voice)
            "-ac", "1",   # Mono
            "-ar", "44100",  # 44.1kHz sample rate
            "-y",  # Overwrite output file
            "-loglevel", "error",  # Only show errors
            compressed_path
        ]

        logger.info(f"Compressing {Path(input_path).name} to MP3...")
        subprocess.run(cmd, check=True, capture_output=True)

        # Log size reduction
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(compressed_path)
        reduction = (1 - compressed_size / original_size) * 100
        logger.info(f"✓ Compressed: {original_size/(1024*1024):.1f}MB → {compressed_size/(1024*1024):.1f}MB ({reduction:.0f}% reduction)")

        return compressed_path

    def upload_file_threaded(self, file_path: str, file_hash: str) -> threading.Thread:
        """
        Compress and upload file to S3 in a background thread using SHA256 hash for organization

        Args:
            file_path: Local path to file
            file_hash: SHA256 hash of the file

        Returns:
            Thread object that is uploading the file
        """
        # Use first 3 characters of hash for subfolder to avoid too many files in one folder
        # With 3 hex chars, we get 4096 possible subfolders (16^3)
        subfolder = file_hash[:3]
        filename = Path(file_path).name
        # Use .mp3 extension since we'll compress to MP3
        base_name = Path(filename).stem
        object_name = f"{subfolder}/{file_hash}_{base_name}.mp3"

        def upload_task():
            compressed_path = None
            upload_path = file_path
            upload_object_name = object_name

            try:
                # Try to compress audio (this happens in background, not blocking GPU)
                try:
                    compressed_path = self.compress_audio(file_path)
                    upload_path = compressed_path
                except Exception as compress_error:
                    logger.warning(f"Compression failed: {compress_error}")
                    logger.info("Falling back to uploading original file")
                    # Use original filename extension for fallback
                    original_ext = Path(filename).suffix
                    upload_object_name = f"{subfolder}/{file_hash}_{Path(filename).stem}{original_ext}"
                    upload_path = file_path

                logger.info(f"Uploading {Path(upload_path).name} to S3 as {upload_object_name}...")
                self.s3_client.upload_file(upload_path, self.bucket, upload_object_name)
                url = f"{self.s3_client.meta.endpoint_url}/{self.bucket}/{upload_object_name}"
                logger.info(f"✓ Uploaded to S3: {url}")

                # Delete both original and compressed files after successful upload
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"✓ Deleted original file: {file_path}")
                if compressed_path and os.path.exists(compressed_path):
                    os.remove(compressed_path)
                    logger.info(f"✓ Deleted compressed file: {compressed_path}")
            except Exception as e:
                logger.error(f"✗ Error uploading {filename} to S3: {e}")
                # Clean up compressed file on error
                if compressed_path and os.path.exists(compressed_path):
                    try:
                        os.remove(compressed_path)
                    except:
                        pass

        thread = threading.Thread(target=upload_task, daemon=True)
        thread.start()
        return thread


class PodcastPileWorker:
    """Worker that fetches jobs and processes them"""

    def __init__(
        self,
        manager_url: str,
        worker_id: str,
        worker_password: Optional[str] = None,
        config: str = "high_latency",
        model_path: Optional[str] = None,
        gpu_id: Optional[int] = None,
        languages: str = "en",
        batch_size: int = 4,
        s3_config: Optional[Dict] = None,
    ):
        """
        Initialize worker

        Args:
            manager_url: URL of the manager server
            worker_id: Unique identifier for this worker
            worker_password: Optional password for worker authentication
            config: Diarization config (very_high_latency, high_latency, low_latency, ultra_low_latency)
            model_path: Optional path to custom .nemo model file
            gpu_id: GPU device ID to use (None for auto-select)
            languages: Comma-separated language codes worker will process
            batch_size: Batch size for FireRedASR transcription (default: 4)
            s3_config: Optional S3 configuration for audio uploads
        """
        self.manager_url = manager_url.rstrip("/")
        self.worker_id = worker_id
        self.worker_password = worker_password
        self.gpu_id = gpu_id
        self.processor = AudioProcessor(
            config=config,
            model_path=model_path,
            gpu_id=gpu_id,
            languages=languages,
            batch_size=batch_size,
        )

        # Initialize S3 uploader if config provided
        self.s3_uploader = S3Uploader(s3_config) if s3_config else None

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
        params = {"worker_id": self.worker_id, "languages": languages}

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
        if url.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
            ext = Path(url).suffix

        filepath = os.path.join(temp_dir, f"audio{ext}")

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"✓ Downloaded to {filepath}")
        return filepath

    def update_job_status(self, job_id: int, status: str) -> bool:
        """Update job status on manager"""
        # Use the /start endpoint when marking as processing
        if status == "processing":
            url = f"{self.manager_url}/api/jobs/{job_id}/start"
            params = {"worker_id": self.worker_id}
        else:
            # For other statuses, we don't have a generic endpoint
            # Just skip the update since complete/fail have their own endpoints
            return True

        try:
            response = requests.post(url, params=params, headers=self.headers)
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
            "processed_at": results.get("processed_at"),
        }

        params = {"worker_id": self.worker_id}

        try:
            response = requests.post(
                url, params=params, json=payload, headers=self.headers
            )
            response.raise_for_status()
            logger.info(f"✓ Submitted results for job #{job_id}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Error submitting results: {e}")
            try:
                logger.error(
                    f"Response body: {e.response.text if e.response else 'N/A'}"
                )
            except:
                pass
            return False

    def report_failure(self, job_id: int, error_message: str) -> bool:
        """Report job failure to manager"""
        url = f"{self.manager_url}/api/jobs/{job_id}/fail"
        params = {"worker_id": self.worker_id, "error_message": error_message}

        try:
            response = requests.post(url, params=params, headers=self.headers)
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

    def process_job(self, job: Dict, next_job: Optional[Dict] = None) -> bool:
        """
        Process a single job, optionally prefetching next job's audio

        Args:
            job: Job dict from manager
            next_job: Optional next job to prefetch audio for

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
        upload_thread = None

        # Prefetch state for next job
        next_temp_dir = None
        next_audio_path = None
        next_download_thread = None

        try:
            # Check if audio was prefetched
            if job.get("_prefetch_audio_path"):
                # Wait for prefetch to complete
                prefetch_thread = job.get("_prefetch_thread")
                if prefetch_thread and prefetch_thread.is_alive():
                    logger.info("Waiting for prefetched download to complete...")
                    prefetch_thread.join()

                prefetch_data = job.get("_prefetch_audio_path")
                temp_dir = job.get("_prefetch_temp_dir")

                # Handle both tuple (new format) and string (old format for backward compatibility)
                if isinstance(prefetch_data, tuple):
                    original_audio_path, audio_path = prefetch_data
                else:
                    # Old format: single path (both are the same)
                    original_audio_path = prefetch_data
                    audio_path = prefetch_data

                if audio_path and os.path.exists(audio_path):
                    logger.info(f"✓ Using prefetched audio: {audio_path}")
                else:
                    # Prefetch failed, download normally
                    logger.warning("Prefetch failed, downloading audio now...")
                    temp_dir = tempfile.mkdtemp()
                    audio_path = self.download_audio(episode_url, temp_dir)
                    original_audio_path = audio_path
            else:
                # No prefetch, download normally
                audio_path = self.download_audio(episode_url, temp_dir)
                original_audio_path = audio_path

            # Start S3 upload in background if configured
            if self.s3_uploader:
                # Compute hash for the ORIGINAL audio file (not converted)
                file_hash = self.processor.compute_file_hashes(original_audio_path)["sha256"]
                upload_thread = self.s3_uploader.upload_file_threaded(
                    original_audio_path, file_hash
                )
                logger.info("S3 upload started in background thread")

            # Process audio with metadata (GPU work happens here)
            # Pass original path for hashing and converted path for processing (if available from prefetch)
            converted_path = audio_path if audio_path != original_audio_path else None
            results = self.processor.diarize_audio(
                original_audio_path,
                episode_url=episode_url,
                language=language,
                converted_audio_path=converted_path
            )

            # Start downloading next job's audio BEFORE submitting results
            # This overlaps download I/O with result submission network I/O
            if next_job:
                next_temp_dir = tempfile.mkdtemp()
                next_episode_url = next_job["episode_url"]

                def download_next():
                    nonlocal next_audio_path
                    try:
                        logger.info(f"⏩ Prefetching audio for job #{next_job['job_id']}...")
                        original_audio_path = self.download_audio(next_episode_url, next_temp_dir)
                        logger.info(f"✓ Prefetched audio for job #{next_job['job_id']}")

                        # Pre-convert to mono if needed (CPU work during I/O time)
                        converted_audio_path = self.processor.convert_to_mono_if_needed(original_audio_path)
                        logger.info(f"✓ Pre-converted audio to mono if needed")

                        # Store both paths: we need the converted for processing, original for hashing
                        next_audio_path = (original_audio_path, converted_audio_path)
                    except Exception as e:
                        logger.warning(f"Failed to prefetch audio: {e}")

                next_download_thread = threading.Thread(target=download_next, daemon=True)
                next_download_thread.start()

            # Submit results (network I/O happens here)
            success = self.submit_results(job_id, results)

            # Wait for S3 upload to complete before finishing
            if upload_thread:
                logger.info("Waiting for S3 upload to complete...")
                upload_thread.join()
                logger.info("S3 upload thread finished")

            # Store prefetch results for next iteration
            if next_download_thread:
                # Attach prefetch info to the job object for next iteration
                next_job["_prefetch_temp_dir"] = next_temp_dir
                next_job["_prefetch_audio_path"] = next_audio_path
                next_job["_prefetch_thread"] = next_download_thread

            return success

        except Exception as e:
            logger.error(f"Error processing job #{job_id}: {e}", exc_info=True)
            self.report_failure(job_id, str(e))
            # Clean up next job prefetch on error
            if next_temp_dir:
                import shutil
                shutil.rmtree(next_temp_dir, ignore_errors=True)
            return False

        finally:
            # Cleanup temp directory (but audio file may already be deleted by S3 uploader)
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
        Continuously request and process jobs with prefetching

        Args:
            languages: Comma-separated language codes (default: "en")
            poll_interval: Seconds to wait between requests if no job available
        """
        import time

        logger.info(f"Starting worker loop (languages: {languages})...")
        logger.info("Press Ctrl+C to stop")

        next_job = None
        prefetch_thread = None

        def prefetch_job():
            """Fetch next job in background"""
            return self.request_job(languages=languages)

        try:
            while True:
                # Use prefetched job if available, otherwise fetch now
                if next_job:
                    job = next_job
                    logger.info(f"Using prefetched job #{job['job_id']}")
                    next_job = None
                else:
                    job = self.request_job(languages=languages)

                if job:
                    # Start prefetching next job in background thread
                    # This overlaps network I/O with GPU processing
                    prefetch_thread = threading.Thread(target=lambda: None, daemon=True)

                    def fetch_wrapper():
                        nonlocal next_job
                        next_job = prefetch_job()
                        if next_job:
                            logger.info(f"✓ Prefetched job #{next_job['job_id']}")

                    prefetch_thread = threading.Thread(target=fetch_wrapper, daemon=True)
                    prefetch_thread.start()

                    # Wait for prefetch to complete before processing
                    # This ensures we can start downloading next audio ASAP
                    if prefetch_thread:
                        prefetch_thread.join(timeout=5)  # Don't wait forever

                    # Process current job, passing next job for audio prefetching
                    self.process_job(job, next_job=next_job)
                else:
                    logger.info(f"No jobs available, waiting {poll_interval}s...")
                    time.sleep(poll_interval)
                    next_job = None  # Clear any stale prefetch

        except KeyboardInterrupt:
            logger.info("\nStopping worker...")
