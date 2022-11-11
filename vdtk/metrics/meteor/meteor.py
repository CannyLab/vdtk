#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

import os
import subprocess
import threading
from typing import Dict, List, Tuple, TypeVar

from jdk4py import JAVA

from vdtk.metrics.corenlp import CORENLP_JAVA_LIBDIR

METEOR_JAR = os.path.join(CORENLP_JAVA_LIBDIR, "lib", "Meteor-1.5.jar")

T = TypeVar("T")


class Meteor:
    def __init__(self) -> None:
        self.meteor_cmd = [str(JAVA), "-jar", "-Xmx8G", METEOR_JAR, "-", "-", "-stdio", "-l", "en", "-norm"]

    def compute_score(self, gts: Dict[T, List[str]], res: Dict[T, List[str]]) -> Tuple[float, Dict[T, float]]:
        # assert (set(gts.keys()) == set(res.keys()))
        imgIds = list(res.keys())

        # Build the score evaluation line
        meteor_input = ""
        for i in imgIds:
            hypothesis_str, reference_list = (res[i][0], gts[i])
            hypothesis_str = hypothesis_str.replace("|||", "").replace("  ", " ")
            score_line = " ||| ".join(("SCORE", " ||| ".join(reference_list), hypothesis_str))
            meteor_input += "{}\n".format(score_line)

        # Get the meteor output
        proc = subprocess.Popen(
            self.meteor_cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        outs, _ = proc.communicate(input=meteor_input.encode("utf-8"))
        proc.terminate()

        ostr = outs.decode("utf-8")
        eval_input = "EVAL |||" + " ||| ".join(ostr.strip().split("\n")) + "\n"

        proc = subprocess.Popen(
            self.meteor_cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        outs, _ = proc.communicate(input=eval_input.encode("utf-8"))
        proc.terminate()

        ostr = outs.decode("utf-8")
        scores = [float(f) for f in ostr.strip().split("\n")]

        return scores[-1], dict(zip(imgIds, scores[:-1]))

    def method(self) -> str:
        return "METEOR"


class MeteorBase:
    def __init__(self) -> None:
        self.meteor_cmd = [str(JAVA), "-jar", "-Xmx2G", METEOR_JAR, "-", "-", "-stdio", "-l", "en", "-norm"]
        self.meteor_p = subprocess.Popen(
            self.meteor_cmd,
            cwd=os.path.dirname(os.path.abspath(METEOR_JAR)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts: Dict[T, List[str]], res: Dict[T, List[str]]) -> Tuple[float, List[float]]:
        assert gts.keys() == res.keys()
        imgIds = gts.keys()
        scores = []

        eval_line = "EVAL"
        self.lock.acquire()
        for i in imgIds:
            assert len(res[i]) == 1
            stat = self._stat(res[i][0], gts[i])
            eval_line += " ||| {}".format(stat)

        assert self.meteor_p.stdin is not None, "Meteor process has no stdin"
        assert self.meteor_p.stdout is not None, "Meteor Process has no stdout"

        self.meteor_p.stdin.write("{}\n".format(eval_line).encode())
        self.meteor_p.stdin.flush()
        for _ in range(0, len(imgIds)):
            scores.append(float(self.meteor_p.stdout.readline().strip()))
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()

        return score, scores

    def method(self) -> str:
        return "METEOR"

    def _stat(self, hypothesis_str: str, reference_list: List[str]) -> str:
        assert self.meteor_p.stdin is not None, "Meteor process has no stdin"
        assert self.meteor_p.stdout is not None, "Meteor Process has no stdout"

        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace("|||", "").replace("  ", " ")
        score_line = " ||| ".join(("SCORE", " ||| ".join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write("{}\n".format(score_line).encode())
        self.meteor_p.stdin.flush()
        return self.meteor_p.stdout.readline().decode().strip()

    def _score(self, hypothesis_str: str, reference_list: List[str]) -> float:
        assert self.meteor_p.stdin is not None, "Meteor process has no stdin"
        assert self.meteor_p.stdout is not None, "Meteor Process has no stdout"

        self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace("|||", "").replace("  ", " ")
        score_line = " ||| ".join(("SCORE", " ||| ".join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write("{}\n".format(score_line).encode("utf-8"))
        stats = self.meteor_p.stdout.readline().strip()
        eval_line = "EVAL ||| {}".format(stats.decode("utf-8"))
        # EVAL ||| stats
        self.meteor_p.stdin.write("{}\n".format(eval_line).encode("utf-8"))
        score = float(self.meteor_p.stdout.readline().strip())
        # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
        # thanks for Andrej for pointing this out
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()
        return score

    def __del__(self) -> None:
        if self.meteor_p.poll() is not None:
            assert self.meteor_p.stdin is not None, "Meteor process has no stdin"
            assert self.meteor_p.stdout is not None, "Meteor Process has no stdout"

            self.lock.acquire()
            self.meteor_p.stdin.close()
            self.meteor_p.kill()
            self.meteor_p.wait()
            self.lock.release()
