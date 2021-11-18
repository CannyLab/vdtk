import logging
import warnings

import click
from rich.logging import RichHandler

from harmony.concept_metrics import concept_overlap
from harmony.core_set import coreset
from harmony.ngram_metrics import ngram_stats
from harmony.vocab_metrics import vocab_stats

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

if __name__ == "__main__":
    cli()
