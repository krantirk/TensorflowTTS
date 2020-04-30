import librosa
import numpy as np
from utils.config_loader import ConfigLoader

# Create a `ConfigLoader` object using a config file and restore a checkpoint or directly load a weights file
config_loader = ConfigLoader('/Users/cschaefe/ttts_weights/standard_config.yaml')
model = config_loader.get_model()
model.load_checkpoint('/Users/cschaefe/ttts_weights/')
# model.load_checkpoint('/path/to/checkpoint/weights/', checkpoint_path=None) # optional: specify checkpoint file
# Run predictions
out = model.predict('Thursday, via a joint press release and Microsoft '
                    'AI Blog, we will announce Microsoft’s continued partnership with Shell '
                    'leveraging cloud, AI, and collaboration technology to drive industry '
                    'innovation and transformation.', encode=True, max_length=2000)

# Convert spectrogram to wav (with griffin lim) and display
stft = librosa.feature.inverse.mel_to_stft(np.exp(out['mel'].numpy().T), sr=22050, n_fft=1024, power=1, fmin=0, fmax=8000)
wav = librosa.feature.inverse.griffinlim(stft, n_iter=32, hop_length=256, win_length=1024)

librosa.output.write_wav('/tmp/sample.wav', wav, sr=22050)
