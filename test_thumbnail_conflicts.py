#!/usr/bin/env python3
"""
Test script to check for thumbnail naming conflicts
"""

import os
from pathlib import Path
import re

def slugify_for_id(value: str) -> str:
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    value = re.sub(r'[-\s]+', '-', value)
    if not value or not value[0].isalpha():
        value = 'id-' + value
    return value

def check_thumbnail_conflicts():
    media_dir = Path('media')
    if not media_dir.exists():
        print('Media directory not found')
        return
    
    files = [f for f in media_dir.iterdir() if f.is_file()]
    slugs = {}
    conflicts = []
    
    for f in files:
        slug = slugify_for_id(f.stem)
        thumbnail_name = f"{slug}.jpg"
        
        if slug in slugs:
            conflicts.append({
                'slug': slug,
                'thumbnail': thumbnail_name,
                'files': [slugs[slug], f.name]
            })
            print(f'CONFLICT: {thumbnail_name}')
            print(f'  - {slugs[slug]}')
            print(f'  - {f.name}')
        else:
            slugs[slug] = f.name
    
    print(f'\nSummary:')
    print(f'Total files: {len(files)}')
    print(f'Unique slugs: {len(slugs)}')
    print(f'Conflicts: {len(conflicts)}')
    
    if conflicts:
        print(f'\nThis explains why the same thumbnail is being applied to multiple videos!')
        print(f'The conflicting files are generating the same thumbnail filename.')

if __name__ == "__main__":
    check_thumbnail_conflicts() 