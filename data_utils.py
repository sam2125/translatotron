import random
import numpy as np
import torch
import torch.utils.data
import os

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths, hparams):
        self.inputs = audiopaths[0]
        self.outputs = audiopaths[1]
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        # random.seed(hparams.seed)
        # random.shuffle(self.audiopaths_and_text)

    def get_mel_spec_pair(self, index):
        # separate filename and text
        # lin = self.get_spec(self.outputs[index])
        # mel = self.get_mel(self.inputs[index])
        inputs = self.get_mel(self.inputs[index])
        outputs = self.get_mel(self.outputs[index])

        return (inputs,outputs)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            # if sampling_rate != self.stft.sampling_rate:
            #     raise ValueError("{} {} SR doesn't match target {} SR".format(
            #         sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_spec(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            spec = self.stft.spectrogram(audio_norm)
            spec = torch.squeeze(spec, 0)
        else:
            spec = torch.from_numpy(np.load(filename))
            # assert melspec.size(0) == self.stft.n_mel_channels, (
            #     'Mel dimension mismatch: given {}, expected {}'.format(
            #         melspec.size(0), self.stft.n_mel_channels))

        return spec

    # def get_text(self, text):
    #     text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
    #     return text_norm

    def __getitem__(self, index):
        return self.get_mel_spec_pair(index)

    def __len__(self):
        return len(self.inputs)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """

        # Right zero-pad mel-spec
        # num_mels = batch[0][1].size(0)
        # max_input_len = max([x[0].size(1) for x in batch])
        # if max_input_len % self.n_frames_per_step != 0:
        #     max_input_len += self.n_frames_per_step - max_input_len % self.n_frames_per_step
        #     assert max_input_len % self.n_frames_per_step == 0

        # # include mel padded and gate padded
        # mel_padded = torch.FloatTensor(len(batch), num_mels, max_input_len)
        # mel_padded.zero_()
        # gate_padded = torch.FloatTensor(len(batch), max_input_len)
        # gate_padded.zero_()
        # input_lengths = torch.LongTensor(len(batch))
        # for i in range(len(batch)):
        #     mel = batch[i][0]
        #     mel_padded[i, :, :mel.size(1)] = mel
        #     gate_padded[i, mel.size(1)-1:] = 1
        #     input_lengths[i] = mel.size(1)

        # # input_lengths, ids_sorted_decreasing = torch.sort(
        # #     torch.LongTensor([len(x[0]) for x in batch]),
        # #     dim=0, descending=True)
        # num_dims = batch[0][1].size(0)
        # max_target_len = max([x[1].size(1) for x in batch])
        # spec_padded = torch.FloatTensor(len(batch), num_dims, max_target_len)
        # spec_padded.zero_()
        # # gate_padded = torch.FloatTensor(len(batch), max_target_len)
        # # gate_padded.zero_()
        # output_lengths = torch.LongTensor(len(batch))
        # for i in range(len(batch)):
        #     spec = batch[i][1]
        #     spec_padded[i, :, :spec.size(1)] = spec
        #     # gate_padded[i, mel.size(1)-1:] = 1
        #     output_lengths[i] = spec.size(1)

        # return mel_padded, gate_padded,input_lengths, spec_padded, \
        #     output_lengths
        num_mels = batch[0][0].size(0)
        # max_input_len = max([x[0].size(1) for x in batch])
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]
        if max_input_len % self.n_frames_per_step != 0:
            max_input_len += self.n_frames_per_step - max_input_len % self.n_frames_per_step
            assert max_input_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        input_padded = torch.FloatTensor(len(batch), num_mels, max_input_len)
        input_padded.zero_()
        # gate_padded = torch.FloatTensor(len(batch), max_target_len)
        # gate_padded.zero_()
        for i in ids_sorted_decreasing:
            mel = batch[i][0]
            input_padded[i, :, :mel.size(1)] = mel
            # gate_padded[i, mel.size(1)-1:] = 1


        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in ids_sorted_decreasing:
            mel = batch[i][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return input_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths
