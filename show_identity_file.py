#!/usr/bin/env python3
"""Show head and tail of identity_conversations.jsonl"""
import os
import json
from nanochat.common import get_base_dir

base_dir = get_base_dir()
path = os.path.join(base_dir, "identity_conversations.jsonl")

print(f"File: {path}")
print(f"Exists: {os.path.exists(path)}")
print()

if not os.path.exists(path):
    print("File does not exist!")
    exit(1)

with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total conversations: {len(lines)}")
print()

print("="*70)
print("HEAD - First 3 conversations:")
print("="*70)
for i, line in enumerate(lines[:3]):
    print(f"\nConversation {i+1}:")
    print("-"*70)
    doc = json.loads(line)
    for msg in doc:
        print(f"{msg['role'].upper()}: {msg['content']}")
    print()

print("="*70)
print("TAIL - Last 3 conversations:")
print("="*70)
for i, line in enumerate(lines[-3:]):
    conv_num = len(lines) - 2 + i
    print(f"\nConversation {conv_num}:")
    print("-"*70)
    doc = json.loads(line)
    for msg in doc:
        print(f"{msg['role'].upper()}: {msg['content']}")
    print()

