import os
import json
import requests
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
import requests

model = InstructBlipForConditionalGeneration.from_pretrained(“Salesforce/instructblip-vicuna-7b”)
processor = InstructBlipProcessor.from_pretrained(“Salesforce/instructblip-vicuna-7b”)
device = “cuda” if torch.cuda.is_available() else "cpu"
model.to(device)
folder_path = “/edrive2/jjulin/data/full_0001_30k/ours_30000”
gt_files = sorted(os.listdir(os.path.join(folder_path, ‘gt’)))
render_files = sorted(os.listdir(os.path.join(folder_path, ‘renders’)))
with open(f'{folder_path}/prompt.json', 'w') as f:
    for i, (gt_file, render_file) in enumerate(zip(gt_files, render_files)):
        if gt_file.endswith('.png') and render_file.endswith('.png'):
            source_path = os.path.join(folder_path, ‘renders’, render_file)
            target_path = os.path.join(folder_path, ‘gt’, gt_file)
            image = Image.open(target_path).resize((256, 256))
            prompt = “describe the image in one sentence but still with plenty of detail.”
            inputs = processor(images=image, text=prompt, return_tensors=“pt”).to(device)
            outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    num_beams=5,
                    max_length=128,
                    min_length=1,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
            )
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            print(generated_text)
            json_entry = {“source”: source_path, “target”: target_path, “prompt”: generated_text}
            f.write(json.dumps(json_entry))
            f.write(‘\n’)
print(“JSON file created successfully!“)