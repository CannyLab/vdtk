from __future__ import division

import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Tuple, TypeVar

import numpy as np
from jdk4py import JAVA

from vdtk.metrics.corenlp import CORENLP_JAVA_LIBDIR

# Assumes spice.jar is in the same directory as spice.py.  Change as needed.
SPICE_JAR = os.path.join(CORENLP_JAVA_LIBDIR, "lib", "*")
TEMP_DIR = "tmp"
CACHE_DIR = "cache"

T = TypeVar("T", str, int)


class Spice:
    """
    Main Class to compute the SPICE metric
    """

    def float_convert(self, obj: Any) -> float:
        try:
            return float(obj)
        except ValueError:
            return np.nan

    def compute_score(
        self, gts: Dict[T, List[str]], res: Dict[T, List[str]]
    ) -> Tuple[float, List[Dict[str, Dict[str, float]]]]:
        assert set(gts.keys()) == set(res.keys())
        imgIds = list(sorted(gts.keys()))

        # Prepare temp input file for the SPICE scorer
        input_data = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert type(hypo) is list
            assert len(hypo) == 1
            assert type(ref) is list
            assert len(ref) >= 1

            input_data.append({"image_id": id, "test": hypo[0], "refs": ref})

        cwd = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(cwd, TEMP_DIR)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        in_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, mode="w+")
        json.dump(input_data, in_file, indent=2)
        in_file.close()

        # Start job
        out_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        out_file.close()
        cache_dir = os.path.join(cwd, CACHE_DIR)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        spice_cmd = [
            str(JAVA),
            "--add-opens",
            "java.base/java.lang=ALL-UNNAMED",
            "--add-opens",
            "java.base/java.math=ALL-UNNAMED",
            "--add-opens",
            "java.base/java.util=ALL-UNNAMED",
            "--add-opens",
            "java.base/java.util.concurrent=ALL-UNNAMED",
            "--add-opens",
            "java.base/java.net=ALL-UNNAMED",
            "--add-opens",
            "java.base/java.text=ALL-UNNAMED",
            "-Xmx8G",
            "-cp",
            SPICE_JAR,
            "edu.anu.spice.SpiceScorer",
            in_file.name,
            "-cache",
            cache_dir,
            "-out",
            out_file.name,
            "-subset",
            "-silent",
        ]
        subprocess.check_call(
            spice_cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )

        # Read and process results
        with open(out_file.name) as data_file:
            results = json.load(data_file)
        os.remove(in_file.name)
        os.remove(out_file.name)

        imgId_to_scores = {}
        spice_scores = []
        for item in results:
            imgId_to_scores[item["image_id"]] = item["scores"]
            spice_scores.append(self.float_convert(item["scores"]["All"]["f"]))
        average_score = np.mean(np.array(spice_scores))
        scores = []
        for image_id in imgIds:
            # Convert none to NaN before saving scores over subcategories
            score_set = {}
            for category, score_tuple in imgId_to_scores[image_id].items():
                score_set[category] = {k: self.float_convert(v) for k, v in score_tuple.items()}
            scores.append(score_set)

        return float(average_score), scores

    def method(self) -> str:
        return "SPICE"
