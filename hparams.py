# import tensorflow as tf
from text import symbols

class mapDict(dict):
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__



def create_hparams(hparams_string=None,verbose=False):
  hparams = {
    ################################
    # Experiment Parameters        #
    ################################
    "epochs":500,
    "iters_per_checkpoint":10,
    "seed":1234,
    "dynamic_loss_scaling":True,
    "fp16_run":False,
    "distributed_run":False,
    "dist_backend":"nccl",
    "dist_url":"tcp://localhost:54321",
    "cudnn_enabled":True,
    "cudnn_benchmark":False,
    "ignore_layers":['embedding.weight'],

    ################################
    # Data Parameters             #
    ################################
    "load_mel_from_disk":False,
    "training_files":'data/train',
    "validation_files":'data/val',
    "text_cleaners":['english_cleaners'],

    ################################
    # Audio Parameters             #
    ################################
    "max_wav_value":32768.0,
    "sampling_rate":22050,
    "filter_length":1024,
    "hop_length":256,
    "win_length":1024,
    "n_mel_channels":80,
    "mel_fmin":0.0,
    "mel_fmax":8000.0,

    #Data parameters
    "input_data_root": '/content/drive/MyDrive/NLP_Project/S2S Parallel data/Hindi_wav',
    "output_data_root": '/content/drive/MyDrive/NLP_Project/S2S Parallel data/Telugu_wav',
    "train_size": 0.99,
    #Output Audio Parameters
    "out_channels":1025,
    ################################
    # Model Parameters             #
    ################################
    "n_symbols":len(symbols),
    "symbols_embedding_dim":512,

    # Encoder parameters
    "encoder_kernel_size":5,
    "encoder_n_convolutions":3,
    "encoder_embedding_dim":128,

    # Decoder parameters
    "n_frames_per_step":1,  # currently only 1 is supported
    "decoder_rnn_dim":256,
    "prenet_dim":32,
    "max_decoder_steps":1000,
    "gate_threshold":0.5,
    "p_attention_dropout":0.1,
    "p_decoder_dropout":0.1,

    # Attention parameters
    "attention_rnn_dim":256,
    "attention_dim":128,
    "attention_heads": 4,

    # Location Layer parameters
    "attention_location_n_filters":32,
    "attention_location_kernel_size":31,

    # Mel-post processing network parameters
    "postnet_embedding_dim":128,
    "postnet_kernel_size":5,
    "postnet_n_convolutions":2,

    ################################
    # Optimization Hyperparameters #
    ################################
    "use_saved_learning_rate":False,
    "learning_rate":1e-3,
    "weight_decay":1e-6,
    "grad_clip_thresh":1.0,
    "batch_size":4,
    "mask_padding":True  
    # set model's padded outputs to padded values
  }

  hparams = mapDict(hparams)


# if hparams_string:
#     tf.logging.info('Parsing command line hparams: %s', hparams_string)
#     hparams.parse(hparams_string)

# if verbose:
#     tf.logging.info('Final parsed hparams: %s', hparams.values())

  return hparams
