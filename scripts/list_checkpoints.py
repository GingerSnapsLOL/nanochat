#!/usr/bin/env python3
"""
List available checkpoints for different training stages.

Usage:
    python -m scripts.list_checkpoints
    python -m scripts.list_checkpoints --source=rl
    python -m scripts.list_checkpoints --source=rl --model-tag=alcoholic_d12
"""

import os
import argparse
import glob
from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import find_largest_model

def list_checkpoints(source="all", model_tag=None):
    """List all available checkpoints for a given source."""
    
    checkpoint_dirs = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }
    
    base_dir = get_base_dir()
    
    if source == "all":
        sources = list(checkpoint_dirs.keys())
    else:
        sources = [source]
    
    for src in sources:
        if src not in checkpoint_dirs:
            print(f"⚠️  Unknown source: {src}")
            continue
            
        checkpoints_dir = os.path.join(base_dir, checkpoint_dirs[src])
        
        if not os.path.exists(checkpoints_dir):
            print(f"\n📁 {src.upper()} checkpoints: {checkpoints_dir}")
            print("   ❌ Directory does not exist")
            continue
        
        print(f"\n📁 {src.upper()} checkpoints: {checkpoints_dir}")
        
        # List all model tags
        model_tags = [f for f in os.listdir(checkpoints_dir) 
                     if os.path.isdir(os.path.join(checkpoints_dir, f))]
        
        if not model_tags:
            print("   ❌ No model tags found")
            continue
        
        # Filter by model_tag if specified
        if model_tag:
            if model_tag not in model_tags:
                print(f"   ❌ Model tag '{model_tag}' not found")
                continue
            model_tags = [model_tag]
        
        for tag in sorted(model_tags):
            model_dir = os.path.join(checkpoints_dir, tag)
            checkpoint_files = glob.glob(os.path.join(model_dir, "model_*.pt"))
            
            if not checkpoint_files:
                print(f"   📦 {tag}: (no checkpoints)")
                continue
            
            # Extract steps and sort
            steps = []
            for f in checkpoint_files:
                try:
                    step = int(os.path.basename(f).split("_")[-1].split(".")[0])
                    steps.append(step)
                except (ValueError, IndexError):
                    continue
            
            if steps:
                steps.sort()
                latest = steps[-1]
                print(f"   📦 {tag}: {len(steps)} checkpoint(s), latest: step {latest:06d}")
                if len(steps) <= 10:
                    print(f"      Steps: {', '.join(f'{s:06d}' for s in steps)}")
                else:
                    print(f"      Steps: {', '.join(f'{s:06d}' for s in steps[:5])} ... {', '.join(f'{s:06d}' for s in steps[-5:])}")
            else:
                print(f"   📦 {tag}: (invalid checkpoint files)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='List available checkpoints')
    parser.add_argument('-s', '--source', type=str, default='all', 
                       choices=['all', 'base', 'mid', 'sft', 'rl'],
                       help='Source to list checkpoints for (default: all)')
    parser.add_argument('-g', '--model-tag', type=str, default=None,
                       help='Filter by specific model tag')
    args = parser.parse_args()
    
    print(f"Base directory: {get_base_dir()}")
    list_checkpoints(args.source, args.model_tag)

