import os
import re
import time
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from pesq import pesq
from pystoi.stoi import stoi
from speechmos import dnsmos
from concurrent.futures import ProcessPoolExecutor, as_completed

# conda activate VocalDiffusionAttack        
# python "7. Audio quality.py" > log.txt 2>&1

# ========== CONFIG ==========
watermarked_path = "../Dataset/Watermarked Audio"
unwatermarked_path = "../Dataset/Unwatermarked Audio"
results_path = "../Dataset/Results"
target_sr = 16000
batch_size = 1000
max_workers = 8  # Ajusta esto según tu CPU
column_names = ["id", "pesq", "stoi", 
                "dnsmos_ovrl_w", "dnsmos_sig_w", "dnsmos_bak_w", "dnsmos_p808_w", 
                "dnsmos_ovrl_un", "dnsmos_sig_un", "dnsmos_bak_un", "dnsmos_p808_un"]
# =============================

def extract_id(filename):
    match = re.search(r'common_voice_en_(\d+)', filename)
    return match.group(1) if match else None

def process_clip(args):
    i, matched, watermarked_path, unwatermarked_path = args
    start = time.time()
    try:
        un, w = matched[i]
        w_path = os.path.join(watermarked_path, w)
        un_path = os.path.join(unwatermarked_path, un)

        w_wav, _ = librosa.load(w_path, sr=target_sr)
        un_wav, _ = librosa.load(un_path, sr=target_sr)

        min_len = min(len(w_wav), len(un_wav))
        w_wav = w_wav[:min_len]
        un_wav = un_wav[:min_len]

        pesq_score = pesq(target_sr, w_wav, un_wav, 'wb')
        stoi_score = stoi(w_wav, un_wav, target_sr, extended=True)
        dnsmos_w_score = dnsmos.run(w_wav, target_sr)
        dnsmos_un_score = dnsmos.run(un_wav, target_sr)

        duration = time.time() - start
        print(f"[{i}] ✔️ {duration:.2f}s", flush=True)

        result = [i, pesq_score, stoi_score]
        result.extend(dnsmos_w_score.values())
        result.extend(dnsmos_un_score.values())
    except Exception as e:
        duration = time.time() - start
        print(f"[{i}] ❌ ERROR ({duration:.2f}s): {e}", flush=True)
        result = [i] + [np.nan] * (len(column_names) - 1)
    return result

if __name__ == "__main__":
    from datetime import datetime
    start_script = datetime.now()

    # Get file lists
    un_files = [f for f in os.listdir(unwatermarked_path) if f.endswith(".mp3") and "audioseal" in f]
    w_files = [f for f in os.listdir(watermarked_path) if f.endswith(".mp3") and "audioseal" in f]

    un_dict = {extract_id(f): f for f in un_files}
    w_dict = {extract_id(f): f for f in w_files}
    matched = {k: (un_dict[k], w_dict[k]) for k in un_dict.keys() & w_dict.keys()}

    print(f"Audios to evaluate: {len(matched):,}")

    # Get already processed IDs
    processed_ids = []
    last_file_num = 0
    for file in os.listdir(results_path):
        if file.startswith("results_audio_quality_") and file.endswith(".csv"):
            file_num = re.search(r'results_audio_quality_(\d+).csv', file)
            if file_num:
                last_file_num = max(last_file_num, int(file_num.group(1)))
                df = pd.read_csv(os.path.join(results_path, file), usecols=["id"])
                processed_ids.extend(df["id"].astype(str).tolist())

    remaining_clips = list(set(matched.keys()) - set(processed_ids))
    print(f"Remaining clips: {len(remaining_clips):,} ({len(remaining_clips)/len(matched):.1%})")
    print(f"Last file number: {last_file_num}")

    args_list = [(i, matched, watermarked_path, unwatermarked_path) for i in remaining_clips]

    # Parallel processing with as_completed
    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_clip, args) for args in args_list]
        for n, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Evaluating clips")):
            result = future.result()
            all_results.append(result)

            if (n + 1) % batch_size == 0 or (n + 1) == len(futures):
                last_file_num += 1
                df = pd.DataFrame(all_results, columns=column_names)
                output_file = f"results_audio_quality_{last_file_num}.csv"
                df.to_csv(os.path.join(results_path, output_file), index=False)
                print(f"\n💾 Saved: {output_file} ({len(all_results)} rows)", flush=True)
                all_results = []

    total_time = datetime.now() - start_script
    print(f"\n✅ DONE in {total_time}")
                