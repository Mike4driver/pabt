#!/usr/bin/env python3
"""
Debug script to see which files are generating problematic thumbnail names
"""

import os
from pathlib import Path
import re

def slugify_for_id(value: str) -> str:
    # Remove characters that are not alphanumeric, underscores, or hyphens
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    # Replace whitespace and sequences of hyphens with a single hyphen
    value = re.sub(r'[-\s]+', '-', value)
    # Ensure it starts with a letter (important for CSS IDs)
    if not value or not value[0].isalpha():
        value = "id-" + value
    return value

def debug_thumbnail_names():
    media_dir = Path('media')
    if not media_dir.exists():
        print('Media directory not found')
        return
    
    files = [f for f in media_dir.iterdir() if f.is_file()]
    
    print("File name -> Slug -> Thumbnail name")
    print("=" * 60)
    
    problematic_files = []
    
    for f in files:
        slug = slugify_for_id(f.stem)
        thumbnail_name = f"{slug}.jpg"
        
        print(f"{f.name} -> {slug} -> {thumbnail_name}")
        
        # Check for problematic patterns
        if slug.startswith('id--') or slug == 'id-' or '--' in slug:
            problematic_files.append((f.name, slug, thumbnail_name))
    
    if problematic_files:
        print(f"\nProblematic files found:")
        print("=" * 60)
        for filename, slug, thumbnail in problematic_files:
            print(f"File: {filename}")
            print(f"Slug: {slug}")
            print(f"Thumbnail: {thumbnail}")
            print()

if __name__ == "__main__":
    debug_thumbnail_names() 