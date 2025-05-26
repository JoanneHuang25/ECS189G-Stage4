import os, gzip, pickle, re
from pathlib import Path
from tqdm import tqdm
from collections import Counter

def clean(text):
    return re.findall(r"[a-z']+", text.lower())

def process_dataset(src_root, out_file, min_freq=2):
    src_root = Path(src_root)
    data = {"train": {"pos": [], "neg": []},
            "test": {"pos": [], "neg": []}}

    vocab_counter = Counter()

    total_files = sum(len(list((src_root / split / label).glob("*.txt"))) 
                     for split in ["train", "test"] 
                     for label in ["pos", "neg"])
    
    with tqdm(total=total_files, desc="Processing files") as pbar:
        for split in ("train", "test"):
            for label in ("pos", "neg"):
                folder = src_root / split / label
                for fname in folder.glob("*.txt"):
                    txt = fname.read_text(encoding="utf8", errors="ignore")
                    tokens = clean(txt)
                    data[split][label].append(tokens)  # Store tokenized text
                    if split == "train":
                        vocab_counter.update(tokens)  # only train data builds vocab
                    pbar.update(1)

    # Build shared vocab
    vocab_tokens = [t for t, c in vocab_counter.items() if c >= min_freq]
    itos = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"] + vocab_tokens
    stoi = {t: i for i, t in enumerate(itos)}
    vocab = {"itos": itos, "stoi": stoi}

    # Save vocab and tokenized data
    packed = {"data": data, "vocab": vocab}

    with gzip.open(out_file, "wb") as f:
        pickle.dump(packed, f)
    print(f"\nSaved to → {out_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process IMDB dataset with shared vocab")
    parser.add_argument("--src", type=str, default="./dataset",
                       help="IMDB dataset directory")
    parser.add_argument("--out", type=str, default="imdb_packed.pkl.gz",
                       help="Output pickle file")
    args = parser.parse_args()
    process_dataset(args.src, args.out)
