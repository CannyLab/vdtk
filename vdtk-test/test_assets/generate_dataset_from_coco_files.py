import json

with open("/home/davidchan/Projects/vdtk/vdtk-test/test_assets/captions_val2014.json", "r") as jf:
    data = json.load(jf)

with open("/home/davidchan/Projects/vdtk/vdtk-test/test_assets/captions_val2014_fakecap_results.json", "r") as jf:
    results = json.load(jf)


# Build the dataset
dataset = {}
for elem in results:
    dataset[elem["image_id"]] = {
        "_id": elem["image_id"],
        "split": "validate",
        "candidates": [elem["caption"]],
        "media_path": "",
        "references": [],
    }

for elem in data["images"]:
    if elem["id"] in dataset:
        dataset[elem["id"]]["media_path"] = elem["file_name"]

for elem in data["annotations"]:
    if elem["image_id"] in dataset:
        dataset[elem["image_id"]]["references"].append(elem["caption"])

# Write the dataset
with open("/home/davidchan/Projects/vdtk/vdtk-test/test_assets/captions_val2014_fakecap_dataset.json", "w") as jf:
    json.dump(list(dataset.values()), jf)
