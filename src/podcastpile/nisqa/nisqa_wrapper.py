# -*- coding: utf-8 -*-
"""
Optimized NISQA wrapper for batch prediction in Podcast Pile
"""
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from . import NISQA_lib as NL


class NISQAPredictor:
    """
    Wrapper for NISQA model optimized for batch predictions.
    Supports both MOS-only and DIM (dimensions) models.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 device: Optional[torch.device] = None,
                 dim: bool = True):
        """
        Initialize NISQA predictor.

        Args:
            model_path: Path to pretrained model (.tar file).
                       If None, uses default weights/nisqa.tar
            device: torch device to use. If None, auto-detects CUDA
            dim: If True, uses DIM model (MOS + dimensions), else MOS only
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim = dim

        # Use default weights if no path provided
        if model_path is None:
            nisqa_dir = Path(__file__).parent
            model_path = nisqa_dir / 'weights' / 'nisqa.tar'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"NISQA model not found at {model_path}")

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.args = checkpoint['args']
        self.args['dim'] = dim

        # Set default sample rate if not in checkpoint (NISQA default is 48kHz)
        if self.args.get('ms_sr') is None:
            self.args['ms_sr'] = 48000

        # Create model
        model_args = {
            'ms_seg_length': self.args['ms_seg_length'],
            'ms_n_mels': self.args['ms_n_mels'],
            'cnn_model': self.args['cnn_model'],
            'cnn_c_out_1': self.args['cnn_c_out_1'],
            'cnn_c_out_2': self.args['cnn_c_out_2'],
            'cnn_c_out_3': self.args['cnn_c_out_3'],
            'cnn_kernel_size': self.args['cnn_kernel_size'],
            'cnn_dropout': self.args['cnn_dropout'],
            'cnn_pool_1': self.args['cnn_pool_1'],
            'cnn_pool_2': self.args['cnn_pool_2'],
            'cnn_pool_3': self.args['cnn_pool_3'],
            'cnn_fc_out_h': self.args['cnn_fc_out_h'],
            'td': self.args['td'],
            'td_sa_d_model': self.args['td_sa_d_model'],
            'td_sa_nhead': self.args['td_sa_nhead'],
            'td_sa_pos_enc': self.args['td_sa_pos_enc'],
            'td_sa_num_layers': self.args['td_sa_num_layers'],
            'td_sa_h': self.args['td_sa_h'],
            'td_sa_dropout': self.args['td_sa_dropout'],
            'td_lstm_h': self.args['td_lstm_h'],
            'td_lstm_num_layers': self.args['td_lstm_num_layers'],
            'td_lstm_dropout': self.args['td_lstm_dropout'],
            'td_lstm_bidirectional': self.args['td_lstm_bidirectional'],
            'td_2': self.args['td_2'],
            'td_2_sa_d_model': self.args['td_2_sa_d_model'],
            'td_2_sa_nhead': self.args['td_2_sa_nhead'],
            'td_2_sa_pos_enc': self.args['td_2_sa_pos_enc'],
            'td_2_sa_num_layers': self.args['td_2_sa_num_layers'],
            'td_2_sa_h': self.args['td_2_sa_h'],
            'td_2_sa_dropout': self.args['td_2_sa_dropout'],
            'td_2_lstm_h': self.args['td_2_lstm_h'],
            'td_2_lstm_num_layers': self.args['td_2_lstm_num_layers'],
            'td_2_lstm_dropout': self.args['td_2_lstm_dropout'],
            'td_2_lstm_bidirectional': self.args['td_2_lstm_bidirectional'],
            'pool': self.args['pool'],
            'pool_att_h': self.args['pool_att_h'],
            'pool_att_dropout': self.args['pool_att_dropout'],
        }

        if dim:
            self.model = NL.NISQA_DIM(**model_args)
        else:
            self.model = NL.NISQA(**model_args)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def predict_files(self,
                     file_paths: List[str],
                     batch_size: int = 8,
                     num_workers: int = 0) -> pd.DataFrame:
        """
        Predict quality scores for a list of audio files.

        Args:
            file_paths: List of paths to audio files
            batch_size: Batch size for inference
            num_workers: Number of workers for data loading

        Returns:
            DataFrame with predictions (mos_pred, and optionally noi_pred,
            dis_pred, col_pred, loud_pred for DIM model)
        """
        # Create dataframe for dataset with dummy values (required by dataset)
        df_data = {
            'filepath_deg': file_paths,
            'mos': [0.0] * len(file_paths)  # Dummy values
        }

        # Add dummy dimension columns if using DIM model
        if self.dim:
            df_data['noi'] = [0.0] * len(file_paths)
            df_data['dis'] = [0.0] * len(file_paths)
            df_data['col'] = [0.0] * len(file_paths)
            df_data['loud'] = [0.0] * len(file_paths)

        df = pd.DataFrame(df_data)

        # Create dataset
        ds = NL.SpeechQualityDataset(
            df=df,
            df_con=None,
            data_dir='',
            filename_column='filepath_deg',
            mos_column=None,
            seg_length=self.args['ms_seg_length'],
            max_length=self.args['ms_max_segments'],
            to_memory=None,
            to_memory_workers=None,
            transform=None,
            ms_n_fft=self.args['ms_n_fft'],
            ms_hop_length=self.args['ms_hop_length'],
            ms_win_length=self.args['ms_win_length'],
            ms_n_mels=self.args['ms_n_mels'],
            ms_sr=self.args['ms_sr'],
            ms_fmax=self.args['ms_fmax'],
            ms_channel=None,
            double_ended=False,
            dim=self.dim,
            filename_column_ref=None,
        )

        # Create dataloader
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers
        )

        # Predict
        with torch.no_grad():
            y_hat_list = []
            for xb, yb, (idx, n_wins) in dl:
                xb = xb.to(self.device)
                n_wins = n_wins.to(self.device)
                y_hat = self.model(xb, n_wins).cpu().numpy()
                y_hat_list.append(y_hat)

        # Concatenate results
        y_hat = np.concatenate(y_hat_list, axis=0)

        # Add predictions to dataframe
        if self.dim:
            ds.df['mos_pred'] = y_hat[:, 0]
            ds.df['noi_pred'] = y_hat[:, 1]
            ds.df['dis_pred'] = y_hat[:, 2]
            ds.df['col_pred'] = y_hat[:, 3]
            ds.df['loud_pred'] = y_hat[:, 4]
        else:
            ds.df['mos_pred'] = y_hat[:, 0]

        return ds.df

    def predict_arrays(self,
                      audio_arrays: List[np.ndarray],
                      sample_rate: int = 16000,
                      batch_size: int = 8) -> Dict[str, np.ndarray]:
        """
        Predict quality scores for audio arrays (using temp files for compatibility).

        Args:
            audio_arrays: List of audio arrays (mono, float32)
            sample_rate: Sample rate of audio (will be resampled if needed)
            batch_size: Batch size for inference

        Returns:
            Dict with keys: 'mos', and optionally 'noisiness', 'discontinuity',
            'coloration', 'loudness' for DIM model
        """
        import tempfile
        import soundfile as sf

        # Resample if needed
        target_sr = int(self.args['ms_sr'])
        if sample_rate != target_sr:
            import librosa
            resampled_arrays = []
            for audio in audio_arrays:
                resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
                resampled_arrays.append(resampled)
            audio_arrays = resampled_arrays
            sample_rate = target_sr

        # Write arrays to temporary WAV files (handle length constraints)
        temp_files = []
        valid_indices = []
        temp_dir = tempfile.mkdtemp()

        # Calculate min and max audio length based on NISQA's requirements
        # seg_length is in spectrogram windows, not seconds
        seg_length = self.args.get('ms_seg_length', 15)  # in windows/frames
        hop_length_sec = self.args.get('ms_hop_length', 0.01)  # in seconds
        # Minimum duration = seg_length * hop_length (e.g., 15 * 0.01 = 0.15 seconds)
        min_duration_sec = seg_length * hop_length_sec
        min_samples = int(min_duration_sec * sample_rate)

        max_segments = self.args.get('ms_max_segments', 1300)
        hop_length_sec = self.args.get('ms_hop_length', 0.01)  # in seconds
        max_duration_sec = max_segments * hop_length_sec  # max duration in seconds
        max_samples = int(max_duration_sec * sample_rate)

        try:
            for i, audio in enumerate(audio_arrays):
                # Skip audio that's too short (NISQA requires min seg_length)
                if len(audio) < min_samples:
                    # Will set to None later
                    continue

                # Trim audio if too long
                if len(audio) > max_samples:
                    audio = audio[:max_samples]

                temp_path = os.path.join(temp_dir, f"temp_{i}.wav")
                sf.write(temp_path, audio, sample_rate)
                temp_files.append(temp_path)
                valid_indices.append(i)

            # Use the file-based prediction method if we have valid files
            if temp_files:
                df = self.predict_files(temp_files, batch_size=batch_size, num_workers=0)
            else:
                df = None

            # Create output arrays with None for all segments initially
            num_arrays = len(audio_arrays)
            if self.dim:
                result = {
                    'mos': np.array([None] * num_arrays, dtype=object),
                    'noisiness': np.array([None] * num_arrays, dtype=object),
                    'discontinuity': np.array([None] * num_arrays, dtype=object),
                    'coloration': np.array([None] * num_arrays, dtype=object),
                    'loudness': np.array([None] * num_arrays, dtype=object),
                }
            else:
                result = {
                    'mos': np.array([None] * num_arrays, dtype=object)
                }

            # Fill in predictions for valid segments
            if df is not None:
                for i, orig_idx in enumerate(valid_indices):
                    if self.dim:
                        result['mos'][orig_idx] = float(df['mos_pred'].iloc[i])
                        result['noisiness'][orig_idx] = float(df['noi_pred'].iloc[i])
                        result['discontinuity'][orig_idx] = float(df['dis_pred'].iloc[i])
                        result['coloration'][orig_idx] = float(df['col_pred'].iloc[i])
                        result['loudness'][orig_idx] = float(df['loud_pred'].iloc[i])
                    else:
                        result['mos'][orig_idx] = float(df['mos_pred'].iloc[i])

            return result
        finally:
            # Clean up temp files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
