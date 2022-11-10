#!/usr/bin/env python
#
# File Name : ptbtokenizer.py
#
# Description : Do the PTB Tokenization and remove punctuations.
#
# Creation Date : 29-12-2014
# Last Modified : Thu Mar 19 09:53:35 2015
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

import os
import sys
import subprocess
import tempfile
import itertools

from jdk4py import JAVA
from vdtk.metrics.corenlp import CORENLP_JAVA_LIBDIR

# path to the stanford corenlp jar
JAR_FILES = os.path.join(CORENLP_JAVA_LIBDIR, "lib", "*")
# JAR_FILE = "stanford-corenlp-3.4.1.jar"

# punctuations to be removed from the sentences
PUNCTUATIONS = [
    "''",
    "'",
    "``",
    "`",
    "-LRB-",
    "-RRB-",
    "-LCB-",
    "-RCB-",
    ".",
    "?",
    "!",
    ",",
    ":",
    "-",
    "--",
    "...",
    ";",
]


class PTBTokenizer:
    """Python wrapper of Stanford PTBTokenizer"""

    def tokenize(self, captions_for_image):
        cmd = [
            str(JAVA),
            "-cp",
            JAR_FILES,
            "edu.stanford.nlp.process.PTBTokenizer",
            "-preserveLines",
            "-lowerCase",
        ]

        # ======================================================
        # prepare data for PTB Tokenizer
        # ======================================================

        final_tokenized_captions_for_image = {}
        image_id = [k for k, v in list(captions_for_image.items()) for _ in range(len(v))]
        sentences = "\n".join([c.replace("\n", " ") for k, v in list(captions_for_image.items()) for c in v])

        # ======================================================
        # save sentences to temporary file
        # ======================================================
        path_to_jar_dirname = os.path.dirname(os.path.abspath(JAR_FILES))
        tmp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        tmp_file.write(sentences)
        tmp_file.close()

        # ======================================================
        # tokenize sentence
        # ======================================================
        cmd.append(tmp_file.name)
        p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        token_lines = p_tokenizer.communicate(input=sentences)[0]
        lines = token_lines.decode("utf-8").split("\n")
        # remove temp file
        os.remove(tmp_file.name)

        # ======================================================
        # create dictionary for tokenized captions
        # ======================================================
        for k, line in zip(image_id, lines):
            if not k in final_tokenized_captions_for_image:
                final_tokenized_captions_for_image[k] = []
            tokenized_caption = " ".join([w for w in line.rstrip().split(" ") if w not in PUNCTUATIONS])
            final_tokenized_captions_for_image[k].append(tokenized_caption)

        return final_tokenized_captions_for_image

    def tokenize_flat(self, captions):
        cmd = [
            str(JAVA),
            "-cp",
            JAR_FILES,
            "edu.stanford.nlp.process.PTBTokenizer",
            "-preserveLines",
            "-lowerCase",
        ]

        # Prepare data for java call
        sentences = "\n".join([c.replace("\n", " ") for c in captions])

        # Save to temporary file
        path_to_jar_dirname = os.path.dirname(os.path.abspath(__file__))
        tmp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        tmp_file.write(sentences)
        tmp_file.close()

        # Tokenize
        cmd.append(tmp_file.name)
        p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        token_lines = p_tokenizer.communicate(input=sentences)[0]
        lines = token_lines.decode("utf-8").split("\n")
        # remove temp file
        os.remove(tmp_file.name)

        # Handle outputs
        tokenized_captions = []
        for line in lines:
            tokenized_caption = " ".join([w for w in line.rstrip().split(" ") if w not in PUNCTUATIONS])
            tokenized_captions.append(tokenized_caption)

        return tokenized_captions
