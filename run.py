import sys
import os
import pandas as pd
from gensim.models import KeyedVectors
import gc
from tqdm import tqdm
import torch
tqdm.pandas()  # Enable tqdm for pandas

# Import từ input directory
sys.path.append('/kaggle/input/embedding-kaggle/KaggleTrain')
import configs
from utils.functions.cpg_utils import parse_to_nodes
from utils import process

# Setup paths - DÙNG BIẾN RIÊNG thay vì override PATHS
INPUT_CPG      = '/kaggle/input/embedding-kaggle/KaggleTrain/data/cpg/'
OUTPUT_INPUT   = '/kaggle/working/input/'
OUTPUT_MODEL   = '/kaggle/working/trained_models/'

# Create output directories
os.makedirs(OUTPUT_INPUT, exist_ok=True)
os.makedirs(OUTPUT_MODEL, exist_ok=True)

def Embed_generator_direct():
    """Read from input, write to working"""
    context = configs.Embed()

    # Get PKL files từ input
    dataset_files = [f for f in os.listdir(INPUT_CPG) if f.endswith('.pkl')]
    print(f"Processing {len(dataset_files)} PKL files...")

    for pkl_file in dataset_files:
        try:
            file_name = pkl_file.split(".")[0]
            print(f"Processing {pkl_file}...")

            # Load từ input - DÙNG ĐƯỜNG DẪN TRỰC TIẾP
            cpg_dataset = pd.read_pickle(os.path.join(INPUT_CPG, pkl_file))

            # Generate embeddings
            print(f"Parsing {len(cpg_dataset)} functions to nodes...")
            cpg_dataset["nodes"] = cpg_dataset.progress_apply(
                lambda row: parse_to_nodes(row.cpg, context.nodes_dim), axis=1
            )

            dummy_kv = KeyedVectors(vector_size=context.nodes_dim)

            print(f"Generating BERT embeddings for {len(cpg_dataset)} functions...")
            cpg_dataset["input"] = cpg_dataset.progress_apply(
                lambda row: process.nodes_to_input(
                    row.nodes, row.target, context.nodes_dim, dummy_kv, context.edge_type
                ),
                axis=1
            )

            # Save input - DÙNG ĐƯỜNG DẪN TRỰC TIẾP
            input_output_path = os.path.join(OUTPUT_INPUT, f"{file_name}_input.pkl")
            cpg_dataset[["input", "target", "func"]].to_pickle(input_output_path)
            
            del cpg_dataset, dummy_kv
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"🧹 Cleaned GPU memory after {file_name}")

        except Exception as e:
            print(f"❌ Error processing {pkl_file}: {e}")
            continue

if __name__ == "__main__":
    Embed_generator_direct()
