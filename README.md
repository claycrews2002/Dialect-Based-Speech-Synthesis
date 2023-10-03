# Dialect-Based-Speech-Synthesis

This is a course project for CSCE 585. We are focusing on the development of a dialect based speech synthesis model.



## Coqui-TTS
[Coqui TTS GitHub Repository](https://github.com/coqui-ai/tts)

### Setup
```
pip install TTS
```

### Voice Conversion
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


Our goal is to generate audio in the dialect of Southern White English(SWE) and African American English(AAE). An audio sample from the podcast 'Southern Fried True Crime' was chosen as the speaker displays characteristics of a SWE speaking dialect. 

Listen to short clip of target speaker, [here](woman_audio_clip.wav)

This audio was used as the target speaker in the voice conversion with the source audio as the previously generated speech example. The `vctk freevc` pre-trained model is an optimal choice for multi-speaker models and voice conversion.

```
tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to("cuda")
tts.voice_conversion_to_file(source_wav="/example_output.wav", target_wav="woman_audio.wav", file_path="conversion_example_output.wav")
```

Listen to voice conversion audio example, [here](conversion_example_output.wav)











