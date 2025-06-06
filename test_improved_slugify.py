#!/usr/bin/env python3
"""
Test the improved slugify function
"""

import hashlib
import re

def slugify_for_id(value: str) -> str:
    original_value = value
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    value = re.sub(r'[-\s]+', '-', value)
    value = value.strip('-')
    if not value or not value[0].isalpha():
        hash_suffix = hashlib.md5(original_value.encode('utf-8')).hexdigest()[:8]
        if value:
            value = f'id-{value}-{hash_suffix}'
        else:
            value = f'id-{hash_suffix}'
    return value

# Test cases
test_files = [
    '♡ NURSE TRY ON HAUL ♡ ︱ LILY DIOR [eKL2gvJAzJA]_2560x1440',
    '123 Video File',
    'Normal Video File',
    '---Special---',
    '',
    '♡♡♡',
    'Video (1)',
    'Video [1]',
    'Video - 1'
]

print("Testing improved slugify function:")
print("=" * 80)

for filename in test_files:
    result = slugify_for_id(filename)
    print(f"'{filename}' -> '{result}' -> '{result}.jpg'")

print("\nThis should fix the thumbnail naming conflicts!") 