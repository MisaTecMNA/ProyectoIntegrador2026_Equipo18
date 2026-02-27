"""
Script 4: Vector Semantics Inspector
Author: Misael
Goal: Check which words align with the steering vector.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# CONFIG
VECTOR_PATH = "/home/tec/code/misael_space/codes/easysteer/data/refusal_vector_2024_injection.pt"
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def inspect_vector_semantics():
    print("Loading model (weights only)...")
    # Load on CPU to avoid using VRAM; this is fast
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"Loading vector: {VECTOR_PATH}")
    vector = torch.load(VECTOR_PATH, map_location="cpu", weights_only=False)
    vector = vector.to(torch.bfloat16)

    # Llama 3 embedding matrix (Unembed)
    # Project the vector against all words in the vocabulary (128k tokens)
    # output_embeddings = model.lm_head.weight  # Shape: [vocab_size, hidden_dim]

    print("Projecting vector into the vocabulary...")
    # Cosine similarity or dot product
    # We use dot product to see direct activation
    unembed_matrix = model.lm_head.weight.detach()

    # logits = vector @ unembed_matrix.T
    scores = torch.matmul(unembed_matrix, vector)

    # Top positive words (what the vector PROMOTES)
    top_scores, top_indices = torch.topk(scores, 20)
    print("TOP 20 words the vector BOOSTS (+):")
    for score, idx in zip(top_scores, top_indices):
        token = tokenizer.decode([idx.item()])
        print(f"   {token:<15} (Score: {score.item():.2f})")

    # Top negative words (what the vector SUPPRESSES)
    bottom_scores, bottom_indices = torch.topk(scores, 20, largest=False)
    print("TOP 20 words the vector SUPPRESSES (-):")
    for score, idx in zip(bottom_scores, bottom_indices):
        token = tokenizer.decode([idx.item()])
        print(f"   {token:<15} (Score: {score.item():.2f})")


if __name__ == "__main__":
    inspect_vector_semantics()
