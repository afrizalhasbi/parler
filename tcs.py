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

# texts =[
        # "Selamat pagi, dengan Sarah dari Telkomsel. Ada yang bisa saya bantu?",
        # "Pagi. Saya mau tanya, kenapa paket internet saya tiba-tiba habis ya?",
        # "Boleh saya tahu nomor HP Bapak/Ibu?",
        # "08123456789",
        # "Baik, bisa saya cek sebentar ya",
        # "Oke",
        # "Menurut sistem, kuota Bapak habis karena penggunaan YouTube cukup tinggi",
        # "Oh gitu... Kalau mau beli paket baru gimana?",
        # "Bapak bisa beli lewat MyTelkomsel atau *363#",
        # "Yang 15GB berapa ya?",
        # "Paket 15GB harganya Rp99.000",
        # "Oke deh makasih infonya",
        # "Sama-sama. Ada yang bisa saya bantu lagi?",
        # "Udah cukup",
        # "Baik, terima kasih sudah menghubungi Telkomsel. Selamat pagi",
    # ]

texts = [
    "Selamat pagi, saya Sarah dari Feedloop. Apa ada yang bisa saya bantu?",
    "Halo, saya ingin menanyakan tentang status pesanan saya.",
    "Tentu, bisa tolong berikan nomor pesanan Anda?",
    "Nomor pesanan saya adalah 123456.",
    "Terima kasih. Saya akan memeriksa status pesanan Anda, mohon tunggu sebentar.",
    "Baik, saya menunggu.",
    "Pesanan Anda saat ini sedang dalam proses pengiriman dan diperkirakan tiba dalam 2 hari.",
    "Oke, apakah saya bisa mendapatkan nomor resi pengiriman?",
    "Tentu, nomor resi pengiriman Anda adalah ABCD1234.",
    "Terima kasih. Apakah ada cara untuk melacak pesanan saya?",
    "Ya, Anda bisa melacaknya melalui situs web penyedia jasa pengiriman dengan memasukkan nomor resi tersebut.",
    "Baik, saya akan coba. Terima kasih atas bantuannya.",
    "Sama-sama! Jika ada pertanyaan lain, jangan ragu untuk menghubungi kami."
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
    sf.write(f"samples_cs/{idx}_{label}.wav", audio_arr, sample_rate)


indah_cond = "Speaker::Indah||Reverb::very close-sounding||Noise::almost no noise||Monotony::slightly expressive and animated||Rate::very slow||Pitch::low-pitch"
tio_cond = "Speaker::Tio||Reverb::very close-sounding||Noise::almost no noise||Monotony::slightly expressive and animated||Rate::very slow||Pitch::low-pitch"
surya_cond = "Speaker::Surya||Reverb::very close-sounding||Noise::almost no noise||Monotony::slightly expressive and animated||Rate::very slow||Pitch::low-pitch"
    
for idx, text in tqdm(enumerate(texts), total=len(texts)):
    # cond = 'Indah' if idx % 2 == 0 else 'Surya'
    # name = 'indah' if idx % 2 == 0 else 'surya'
    cond = 'indah'
    name = 'indah'
    os.makedirs('samples_cs', exist_ok=True)
    generate_audio(cond, text, idx, name)

# for idx, text in tqdm(enumerate(texts), total=len(texts)):
    # os.makedirs('samples', exist_ok=True)
    # generate_audio(surya_cond, text, idx, "surya")
