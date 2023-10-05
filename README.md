# Dialect-Based-Speech-Synthesis

This is a course project for CSCE 585. We are focusing on the development of a dialect based speech synthesis model.

## Project Milestones
- [Proposal](https://github.com/claycrews2002/Dialect-Based-Speech-Synthesis/blob/aad98979f4374c79ca33ff970e02ec136c378eb1/CSCE585%20-%20Wordification%20Proposal.pdf)
- [Milestone 1](https://github.com/claycrews2002/Dialect-Based-Speech-Synthesis/blob/aad98979f4374c79ca33ff970e02ec136c378eb1/CSCE585%20-%20Wordification%20Project%20Milestone%201.pdf)


We found Coqui-TTS as a starting point for our research. This repository was chosen because of the easy access to pre-trained models and a python API to quickly implement our initial testing.

## Coqui-TTS
[Coqui TTS GitHub Repository](https://github.com/coqui-ai/tts)

### Setup
```
pip install TTS
```

### Voice Generation
At this point we are interested in using a pre-trained model to do voice conversion.

This is a simple use case of the TTS functionality to generate an audio file from a pre-trained model.
```
import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS with the target model name
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False).to(device)

# Run TTS
tts.tts_to_file(text="This is an example sentence using the LJ Speech dataset on a pre-trained model using this text to speech generation tool.", file_path="example_output.wav")
```

Listen to generated example audio, [here](example_output.wav)

### Voice Conversion

Our goal is to generate audio in the dialect of Southern White English(SWE) and African American English(AAE). An audio sample from the podcast 'Southern Fried True Crime' was chosen as the speaker displays characteristics of a SWE speaking dialect. 

Listen to short clip of target speaker, [here](woman_audio_clip.wav)

This audio was used as the target speaker in the voice conversion with the source audio as the previously generated speech example. The `vctk freevc` pre-trained model is an optimal choice for multi-speaker models and voice conversion.

```
tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to("cuda")
tts.voice_conversion_to_file(source_wav="/example_output.wav", target_wav="woman_audio.wav", file_path="conversion_example_output.wav")
```

Listen to voice conversion audio example, [here](conversion_example_output.wav)

## Papers and Methods Examined

### Spectogram Models
- [Tacotron](https://arxiv.org/abs/1703.10135)
- [Tacotron2](https://arxiv.org/abs/1712.05884)

### Attention Methods
- [Guided Attention](https://arxiv.org/abs/1710.08969)

### Vocoders
- [MelGAN](https://arxiv.org/abs/1910.06711)
- [WaveRNN](https://github.com/fatchord/WaveRNN/)

### Voice Conversion
- [FreeVC](https://arxiv.org/abs/2210.15418)


## Next Steps

We will continue to read papers on various architectures sourrounding speech synthesis.

The direction we plan to take our research is to create a model that compares to [FreeVC](https://arxiv.org/abs/2210.15418) yielding better results. Different architectures and implementations will be examined to achieve this goal. 

Processing the input data of the target speaker's voice, dialect, and speaking style will be necessary attributes of the audio to draw attention to. 

Creating a more efficient voice conversion model would eliminate the need for long training times and effectively produce clear and accurate results comapred to the target speaker's voice.











