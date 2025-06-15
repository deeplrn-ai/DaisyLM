import os
from tqdm import tqdm
from datasets import load_dataset
import json
import logging
import tiktoken

BASE_DATA_DIR = '/Volumes/Programming/'
PRETRAINING_DATA_DIR = os.path.join(BASE_DATA_DIR, 'training_data')
RAW_DATA_CACHE_DIR = os.path.join(BASE_DATA_DIR, '.cache/huggingface/datasets')

os.makedirs(PRETRAINING_DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_CACHE_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATASETS_TO_PROCESS = [
    {'name': 'openai/gsm8k', 'config_name': 'arxiv', 'split': 'train'},
    {'name': 'nampdn-ai/tiny-lessons', 'config_name': None, 'split': 'train'},
    {'name': 'togethercomputer/RedPajama-Data-1T', 'config_name': 'arxiv', 'split': 'train[:20%]'},
    {'name': 'togethercomputer/RedPajama-Data-1T', 'config_name': 'wikipedia', 'split': 'train[:30%]'},
]

CHUNK_SIZE = 2048
TOKENS_PER_OUTPUT_FILE = 500_000

total_chunks_saved = 0
file_idx_tracker = {}

logging.info('loading tiktoken tokenizer (cl100k base)..')
try:
    tokenizer = tiktoken.get_encoding('cl100k_base')
except Exception as e:
    logging.error(f'Failed to load tiktoken tokenizer: {e}')
    exit(1)

def process_dataset(dataset_info):
    global total_chunks_saved, file_idx_tracker

    dataset_name = dataset_info['name']
    dataset_config_name = dataset_info['config_name']
    dataset_split = dataset_info['split']
    data_files = dataset_info.get("data_files")

    safe_dataset_name = dataset_name.replace('/', '__')
    if dataset_config_name:
        safe_dataset_name = f"{safe_dataset_name}__{dataset_config_name}"

    current_output_dir = os.path.join(PRETRAINING_DATA_DIR, safe_dataset_name)
    os.makedirs(current_output_dir, exist_ok=True)

    if safe_dataset_name not in file_idx_tracker:
        file_idx_tracker[safe_dataset_name] = 0 

    logging.info(f"\n--- Processing Dataset: {dataset_name} ({dataset_config_name or 'default'}) ---")
    logging.info(f"Tokenized output for this dataset will go into: {current_output_dir}")

    load_params = {
        "path": dataset_name,
        "split": dataset_split, 
        "streaming": True,
        "cache_dir": RAW_DATA_CACHE_DIR,
    }
    if dataset_config_name:
        load_params["name"] = dataset_config_name
    if data_files:
        load_params["data_files"] = data_files

    try:
        ds = load_dataset(**load_params)
        logging.info(f"Dataset '{dataset_name}' loaded in streaming mode (split: {dataset_split}).")
    except Exception as e:
        logging.error(f"Failed to load dataset {dataset_name} (Config: {dataset_config_name}): {e}")
        logging.error("This dataset might not be streamable or the path/config_name is incorrect. Please verify on Hugging Face.")
        return

    chunk_buffer_tokens = []
    current_buffer_token_count = 0

    pbar = tqdm(enumerate(ds), desc=f"Processing {dataset_name}")

    for i, item in pbar:
        text = ""

        if dataset_name == "openai/gsm8k":
            question = item.get("question", "")
            answer = item.get("answer", "")
            text = f"Question: {question}\nAnswer: {answer}"
        elif dataset_name == "nampdn-ai/tiny-lessons":
            text = item.get("text", "")
            if not text:
                text = item.get("lesson_content", "")
        elif dataset_name == "togethercomputer/RedPajama-Data-1T":
                text = item.get("text", "")
        else:
            text = item.get("text", "")

        if not text:
            pbar.set_postfix_str(f"Skipped empty text item {i} (no content found in expected fields)")
            continue

        try:
            tokens = tokenizer.encode(text)
        except Exception as e:
            logging.warning(f"Skipping item {i} due to tokenization error: {e}")
            continue

        for j in range(0, len(tokens), CHUNK_SIZE):
            chunk = tokens[j : j + CHUNK_SIZE]

            chunk_buffer_tokens.append(chunk)
            current_buffer_token_count += len(chunk)
            total_chunks_saved += 1

        if current_buffer_token_count >= TOKENS_PER_OUTPUT_FILE:
                save_current_buffer(chunk_buffer_tokens, current_output_dir, safe_dataset_name)
                chunk_buffer_tokens.clear()
                current_buffer_token_count = 0
            
    if chunk_buffer_tokens:
        save_current_buffer(chunk_buffer_tokens, current_output_dir, safe_dataset_name)
    
    logging.info(f"Finished processing {dataset_name}. Total chunks saved so far: {total_chunks_saved}")

def save_current_buffer(buffer, dataset_output_dir, dataset_key):
    global file_idx_tracker
    if not buffer:
        return
    
    file_idx = file_idx_tracker[dataset_key]
    file_path = os.path.join(dataset_output_dir, f"chunks_{file_idx:06d}.jsonl")
    
    try:
        with open(file_path, "w") as f:
            for token_list in buffer:
                json.dump(token_list, f)
                f.write("\n")
        logging.info(f"Saved {len(buffer)} chunks to: {file_path}")
        file_idx_tracker[dataset_key] += 1
    except IOError as e:
        logging.error(f"Error saving file {file_path}: {e}")

if __name__ == "__main__":
    logging.info("Starting data preprocessing script...")
    logging.info(f"Tokenized data will be saved to subdirectories within: {PRETRAINING_DATA_DIR}")
    logging.info(f"Raw Hugging Face cache will be in: {RAW_DATA_CACHE_DIR}")
    
    try:
        import zstandard
        logging.info("zstandard library detected. Compression should work for RedPajama datasets.")
    except ImportError:
        logging.error("zstandard library not found. Please install it: pip install zstandard")
        exit(1)

    for ds_info in DATASETS_TO_PROCESS:
        process_dataset(ds_info)

    logging.info(f"\n✅ All datasets processed. Total chunks saved across all datasets: {total_chunks_saved}")
    logging.info(f"Final tokenized data is in subdirectories within: {PRETRAINING_DATA_DIR}")
    logging.info(f"Remember to check disk space in '{BASE_DATA_DIR}' for raw dataset cache.")


