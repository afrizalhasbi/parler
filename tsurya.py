from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
import soundfile as sf
import torch
import os
from tqdm import tqdm
import numpy as np
import sys

path = sys.argv[1]
device = "cuda" if torch.cuda.is_available() else "cpu"
token = ""
model = ParlerTTSForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, token=token, ignore_mismatched_sizes=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(path, token=token)

texts = [
    'Sepanjang jalan kenangan. Kita selalu bergandeng tangan',
    'Dengan senang hati, bapak. Saya senang bisa membantu',
    'Mohon maaf, saya izin interupsi',
    'Apa kabar? Selamat siang! Ada yang bisa saya bantu?',
    'Saya suka tempe, tapi tidak suka tahu',
    'Untuk produk itu tidak kami layani, ibu',
    'Saya harus berikan informasi ini segera kepada anda dan mereka semua.',
    'Kucing itu runcing kuku kaki cakarnya.',
    'Nah, itu yang saya maksud dari kemarin.',
    'Kalau kata atasan saya, request bapak itu melanggar prosedur.'
    ]

def generate_audio(description, text, idx, label):
    # Tokenize with padding and attention mask
    inputs = tokenizer(
        description,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True
    )

    prompts = tokenizer(
        text,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True
    )

    # Move everything to device
    input_ids = inputs.input_ids.to(device)
    input_attention_mask = inputs.attention_mask.to(device)
    prompt_input_ids = prompts.input_ids.to(device)
    prompt_attention_mask = prompts.attention_mask.to(device)

    # Generate with attention masks
    generation = model.generate(
        input_ids=input_ids,
        attention_mask=input_attention_mask,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask
    )

    # Process and save audio
    audio_arr = generation.to(torch.float32).cpu().numpy().squeeze().astype(np.float32)
    sample_rate = 44100
    sf.write(f"samples/{label}_{idx}.wav", audio_arr, sample_rate)


indah_cond = "Speaker::Indah||Reverb::very close-sounding||Noise::almost no noise||Monotony::slightly expressive and animated||Rate::slow||Pitch::high-pitch"
tio_cond = "Speaker::Tio||Reverb::very close-sounding||Noise::almost no noise||Monotony::slightly expressive and animated||Rate::slow||Pitch::low-pitch"
surya_cond = "Speaker::Surya||Reverb::very close-sounding||Noise::almost no noise||Monotony::slightly expressive and animated||Rate::slow||Pitch::low-pitch"

# for name in ['indah', 'tio']:
    # cond = indah_cond if name == 'indah' else tio_cond
    # for idx, text in tqdm(enumerate(texts), total=len(texts)):
        # os.makedirs('samples', exist_ok=True)
        # generate_audio(indah_cond, text, idx, name)

for idx, text in tqdm(enumerate(texts), total=len(texts)):
    os.makedirs('samples', exist_ok=True)
    generate_audio(surya_cond, text, idx, "surya")
