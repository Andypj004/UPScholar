"""
Script to pre-download all models
Run during Docker build
"""

from sentence_transformers import SentenceTransformer

models = [
    'all-MiniLM-L6-v2',
    'all-mpnet-base-v2',
    'paraphrase-multilingual-MiniLM-L12-v2',
    'bert-base-nli-mean-tokens'
]

print("="*80)
print("PRE-DOWNLOADING BERT MODELS")
print("="*80)

for i, model_name in enumerate(models, 1):
    print(f"\n[{i}/{len(models)}] Downloading: {model_name}")
    print("-" * 80)
    try:
        model = SentenceTransformer(model_name)
        print(f"✓ {model_name} downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")

print("\n" + "="*80)
print("✓ PROCESS COMPLETED")
print("="*80)