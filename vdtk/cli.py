import logging
import warnings

import click
from rich.logging import RichHandler

from vdtk.caption_metrics import caption_stats
from vdtk.concept_metrics import concept_leave_one_out, concept_overlap
from vdtk.core_set import coreset
from vdtk.leave_one_out import leave_one_out
from vdtk.ngram_metrics import ngram_stats
from vdtk.qualitative_sample import qualitative_sample
from vdtk.semantic_variance import semantic_variance
from vdtk.vocab_metrics import vocab_stats

FORMAT = "%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

# Deal with NLTK being too verbose
warnings.filterwarnings("ignore")


@click.group()
def cli():
    pass


# Add the commands to the CLI
cli.add_command(vocab_stats)
cli.add_command(ngram_stats)
cli.add_command(concept_overlap)
cli.add_command(coreset)
cli.add_command(semantic_variance)
cli.add_command(leave_one_out)
cli.add_command(concept_leave_one_out)
cli.add_command(caption_stats)
cli.add_command(qualitative_sample)

if __name__ == "__main__":
    cli()
