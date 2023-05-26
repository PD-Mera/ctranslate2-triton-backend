import asyncio
import tritonclient.grpc.aio
from tritonclient.utils import np_to_triton_dtype
from grpc import ChannelConnectivity
from transformers import AutoTokenizer
import logging
import numpy as np
import sys
import ctranslate2 
import librosa
import transformers

def main():
    MODEL_NAME = "whisper"
    URL = "10.9.3.239:8388"
    client = tritonclient.grpc.InferenceServerClient(URL)
    # Load and resample the audio file.
    audio, _ = librosa.load(R"D:\AI\iCOMM\Triton\segment_1961773_0.wav", sr=16000, mono=True)
    # en_text = sys.stdin.readline()
    processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-tiny")
    inputs = processor(audio, return_tensors="np", sampling_rate=16000)
    # features = inputs.input_features
    features = ctranslate2.StorageView.from_array(inputs.input_features)
    features1 = np.array(inputs.input_features).reshape((1, 240000))
    # features1 = features1
    print(features1)
    print(features1.shape)
    language = 'vi'
    prompt = processor.tokenizer.convert_tokens_to_ids([
            "<|startoftranscript|>",
            "<|vi|>",
            "<|transcribe|>",
            "<|notimestamps|>",  # Remove this token to generate timestamps.
        ])
    prompt = np.array([prompt,], dtype=np.int32)
    print(prompt)
    # model = ctranslate2.models.Whisper("whisper-tiny-ct2")


    logging.info(f"Tokenised input: {features}")
    
    # if client._channel.get_state() == ChannelConnectivity.SHUTDOWN:
    #     return
    # print(features.DataType)
    inputs = [
        tritonclient.grpc.InferInput("FEATURES", features1.shape, np_to_triton_dtype(np.float32)),
        tritonclient.grpc.InferInput("PROMPTS_IDS", prompt.shape, np_to_triton_dtype(prompt.dtype)),
    ]
    inputs[0].set_data_from_numpy(features1)
    inputs[1].set_data_from_numpy(prompt)
    outputs = [tritonclient.grpc.InferRequestedOutput("OUTPUT_IDS")]

    res = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
    results = res.as_numpy("OUTPUT_IDS")
    print(results)
    logging.info(f"Returned tokens: {res}")
    transcription = processor.decode(results[0])
    print(transcription)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    # asyncio.run(main())




# Load and resample the audio file.audio, _ = librosa.load("audio.wav", sr=16000, mono=True)
# Compute the features of the first 30 seconds of audio.



# Load the model on CPU.model = ctranslate2.models.Whisper("whisper-tiny-ct2")

# Run generation for the 30-second window.
# results = model.generate(features, [prompt])transcription = processor.decode(results[0].sequences_ids[0])
# print(transcription)