import os
import json

folder_path = "jeff_jhu_apl_data/full_0000_30k/ours_30000/"
gt_folder = os.path.join(folder_path, 'gt')
render_folder = os.path.join(folder_path, 'renders')

# Get sorted lists of files in 'gt' and 'renders' folders
gt_files = sorted(f for f in os.listdir(gt_folder) if f.endswith('.png'))
render_files = sorted(f for f in os.listdir(render_folder) if f.endswith('.png'))

# Create and write JSON entries to a file
# json_entries = []
json_dict = {}
json_dict["values"] = []

for gt_file, render_file in zip(gt_files, render_files):
    source_path = os.path.join(render_folder, render_file)
    target_path = os.path.join(gt_folder, gt_file)
    prompt_text = "clean, clear image"
    json_entry = {"source": source_path, "target": target_path, "prompt": prompt_text}
    # json_entries.append(json_entry)
    json_dict["values"].append(json_entry)

output_json_path = os.path.join(folder_path, 'prompt.json')
with open(output_json_path, 'w') as f:
    json.dump(json_dict, f, indent=4)

print("JSON file created successfully at:", output_json_path)
