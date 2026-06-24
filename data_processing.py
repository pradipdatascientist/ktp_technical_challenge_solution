import os
import json
import torch
from datasets import load_dataset
from PIL import Image
import io

def process_dataset(output_dir="processed_data", num_samples=50):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    print(f"Loading dataset small-publaynet-wds...")
    dataset = load_dataset("/small-publaynet-wds", split="train", streaming=True)
    
    data = []
    count = 0
    for sample in dataset:
        if count >= num_samples:
            break
            
        key = sample["__key__"]
        metadata = sample["json"]
        image_data = sample["png"]
        
        # Save image
        image_path = os.path.join(output_dir, "images", f"{key}.png")
        image_data.save(image_path)
        
        # Extract text and regions
        text_content = ""
        regions = []
        for ann in metadata.get("annotations", []):
            category_id = ann.get("category_id")
            # In PubLayNet: 1: text, 2: title, 3: list, 4: table, 5: figure
            category_map = {1: "text", 2: "title", 3: "list", 4: "table", 5: "figure"}
            category = category_map.get(category_id, "unknown")
            
            # Note: The dataset might not have direct text in annotations, 
            # we might need to rely on the metadata or mock some text for this POC
            # if direct text is missing.
            bbox = ann.get("bbox")
            regions.append({
                "category": category,
                "bbox": bbox,
                "id": ann.get("id")
            })
            
        # For the sake of this POC, we'll create some mock text if it's missing
        # in a real scenario, we'd use OCR or the extracted text field.
        # Looking at the dataset card, it seems annotations have segmentation but not raw text.
        # We will simulate text extraction for the entities.
        mock_text = f"This document {key} contains {len(regions)} regions including "
        mock_text += ", ".join([r['category'] for r in regions[:3]]) + "."
        
        data.append({
            "id": key,
            "image_path": image_path,
            "regions": regions,
            "text": mock_text,
            "metadata": metadata
        })
        
        count += 1
        if count % 10 == 0:
            print(f"Processed {count} samples...")

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Data processing complete. Saved {len(data)} samples to {output_dir}")

if __name__ == "__main__":
    process_dataset()
