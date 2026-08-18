import json
from collections import Counter

with open(r'images\MAGFiLO_1.0_Kaggle_2026\train\MAGFiLO_1.0_Annotations_kaggle2026_train.json', 'r') as f:
    d = json.load(f)

print("=== INFO ===")
print(d.get('info', {}))
print("\n=== LICENSES ===")
print(d.get('licenses', []))
print("\n=== CATEGORIES ===")
for c in d['categories']:
    print(f"  {c}")

print("\n=== ANNOTATION STATS ===")
cat_dist = Counter([a['category_id'] for a in d['annotations']])
print(f"Category distribution: {dict(cat_dist)}")

areas = [a['area'] for a in d['annotations']]
print(f"Area stats: min={min(areas)}, max={max(areas)}, mean={sum(areas)/len(areas):.1f}")

has_seg = sum(1 for a in d['annotations'] if a.get('segmentation'))
has_spine = sum(1 for a in d['annotations'] if a.get('spine'))
print(f"Has segmentation: {has_seg}/{len(d['annotations'])}")
print(f"Has spine: {has_spine}/{len(d['annotations'])}")

img_ids = set(a['image_id'] for a in d['annotations'])
print(f"Images with annotations: {len(img_ids)}/{len(d['images'])}")

# Check which train images have annotations
import os
train_imgs = set(os.listdir(r'images\MAGFiLO_1.0_Kaggle_2026\train\train_images'))
anno_filenames = set(img['file_name'] for img in d['images'])
print(f"\nTrain dir images: {len(train_imgs)}")
print(f"Annotation image entries: {len(d['images'])}")
overlap = train_imgs & anno_filenames
print(f"Overlap: {len(overlap)}")

# Image dimensions
from collections import Counter
dims = Counter([(img['width'], img['height']) for img in d['images']])
print(f"\nImage dimensions distribution: {dict(dims)}")

# Filament count per image
filaments_per_img = Counter([a['image_id'] for a in d['annotations']])
vals = list(filaments_per_img.values())
print(f"\nFilaments per image: min={min(vals)}, max={max(vals)}, mean={sum(vals)/len(vals):.1f}")
