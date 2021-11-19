import logging
import warnings

import click
from rich.logging import RichHandler

from harmony.concept_metrics import concept_overlap
from harmony.core_set import coreset
from harmony.leave_one_out import leave_one_out
from harmony.ngram_metrics import ngram_stats
from harmony.semantic_variance import semantic_variance
from harmony.vocab_metrics import vocab_stats

FORMAT = "%(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

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

if __name__ == "__main__":
    cli()
