# vdtk: Visual Description Evaluation Tools

This tool is designed to allow for a deep investigation of diversity in visual description datasets, and to help users
understand their data at a token, n-gram, description, and dataset level.

## Installation

To use this tool, you can easily pip install with `pip install vdtk`.

## Data format

In order to prepare datasets to work with this tool, datasets must be formatted as JSON files with the following schema
```json
// List of samples in the dataset
[
    // JSON object for each sample
    {
        "_id": "string", // A string ID for each sample. This can help keep track of samples during use.
        "split": "string", // A string corresponding to the split of the data. Default splits are "train", "validate" and "test"
        "references": [
            // List of string references
            "reference 1...",
            "reference 2...",
        ],
        "candidates": [
            // List of string candidates (Optional)
            "candidate 1...",
            "candidate 2...",
        ],
        "media_path": "string", // (Optional) Path to the image/video (for image/video based metrics, recall experiemnts, etc.)
        "metadata": {} // Any JSON object. This field is not used by the toolkit at this time.
    }
]
```

## Usage

After installation, the basic menu of commands can be accessed with `vdtk --help`. We make several experiments/tools
available for use:

| Command | Details |
| ----------- | ----------- |
| vocab-stats | Run with `vdtk-cli vocab-stats DATASET_JSON_PATH`. Compute basic token-level vocab statistics |
| ngram-stats | Run with `vdtk-cli ngram-stats DATASET_JSON_PATH`. Compute n-gram statistics, EVS@N and ED@N  |
| caption-stats | Run with `vdtk-cli caption-stats DATASET_JSON_PATH`. Compute caption-level dataset statistics  |
| semantic-variance | Run with `vdtk-cli semantic-variance DATASET_JSON_PATH`. Compute within-sample BERT embedding semantic variance |
| coreset | Run with `vdtk-cli coreset DATASET_JSON_PATH`. Compute the caption coreset from the training split needed to solve the validation split |
| concept-overlap | Run with `vdtk-cli concept-overlap DATASET_JSON_PATH`. Compute the concept overlap between popular feature extractors, and the dataset |
| concept-leave-one-out | Run with `vdtk-cli concept-leave-one-out DATASET_JSON_PATH`. Compute the performance with a coreset of concept captions |
| leave-one-out | Run with `vdtk-cli leave-one-out DATASET_JSON_PATH`. Compute leave-one-out ground truth performance on a dataset with multiple ground truths |

Additionally, several commands take multiple dataset JSONs, which can be used to compare different runs, or different datasets. Appending (:baseline) to any
of the JSON file paths will treat this run as a baseline, and compute relative values and coloring accordingly (example: `vdtk-cli score cider-d ./baseline.json:baseline ./model.json`).

| Command | Details |
| ----------- | ----------- |
| score | Run with `vdtk-cli score [metric] DATASET_JSON_PATH_1, DATASET_JSON_PATH_2...`. Compute BLEU/METEOR/CIDEr-D/ROUGE/BERTScore/MAUVE/etc. Guaranteed to be consistent with the COCO captioning tools (for use externally). |
| clip-recall | Run with `vdtk-cli clip-recall DATASET_JSON_PATH_1, DATASET_JSON_PATH_2...`. Compute the MRR, and Recall@K values for candidate/reference captions based on the CLIP model. |
| content-recall | Run with `vdtk-cli content-recall DATASET_JSON_PATH_1, DATASET_JSON_PATH_2...`. Compute Noun/Verb recall for the candidates against the references. |

For more details and options, see the `--help` command for any of the commands above. Note that some tools are relatively
compute intensive. This toolkit will make use of a GPU if available and necessary, as well as a large number of CPU cores
and RAM depending on the task.
