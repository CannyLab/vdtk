# Harmony: Visual Description Evaluation Data Tools

This tool is designed to allow for a deep investigation of diversity in visual description datasets, and to help users
understand their data at a token, n-gram, description, and dataset level.

## Installation

To use this tool, you can easily pip install with `pip install .` from this directory. Note: Some metrics (METEOR) require
a working installation of Java. Please follow the directions (here) to install the Java runtime if you do not already
have access to a JRE.

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
        "metadata": {} // Any JSON object. This field is not used by the toolkit at this time.
    }
]
```

## Usage

After installation, the basic menu of commands can be accessed with `harmony-cli --help`. We make several experiments/tools
available for use:

| Command | Details |
| ----------- | ----------- |
| vocab-stats | Run with `harmony-cli vocab-stats DATASET_JSON_PATH`. Compute basic token-level vocab statistics |
| ngram-stats | Run with `harmony-cli ngram-stats DATASET_JSON_PATH`. Compute n-gram statistics, EVS@N and ED@N  |
| caption-stats | Run with `harmony-cli caption-stats DATASET_JSON_PATH`. Compute caption-level dataset statistics  |
| semantic-variance | Run with `harmony-cli semantic-variance DATASET_JSON_PATH`. Compute within-sample BERT embedding semantic variance |
| coreset | Run with `harmony-cli coreset DATASET_JSON_PATH`. Compute the caption coreset from the training split needed to solve the validation split |
| concept-overlap | Run with `harmony-cli concept-overlap DATASET_JSON_PATH`. Compute the concept overlap between popular feature extractors, and the dataset |
| concept-leave-one-out | Run with `harmony-cli concept-leave-one-out DATASET_JSON_PATH`. Compute the performance with a coreset of concept captions |
| leave-one-out | Run with `harmony-cli vocab-stats DATASET_JSON_PATH`. Compute leave-one-out ground truth performance on a dataset with multiple ground truths |
| **[BETA]** balanced-split | Run with `harmony-cli balanced-split DATASET_JSON_PATH`. Compute a set of splits of the data which best balance the data diversity |

For more details and options, see the `--help` command for any of the commands above. Note that some tools are relatively
compute intensive. This toolkit will make use of a GPU if available and necessary, as well as a large number of CPU cores
and RAM depending on the task.

**[BETA]** See the [API Docs](https://) for usage as a library.
