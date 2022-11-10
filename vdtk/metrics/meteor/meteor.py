#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

import os
import sys
import subprocess
import threading

from jdk4py import JAVA
from vdtk.metrics.corenlp import CORENLP_JAVA_LIBDIR

METEOR_JAR = os.path.join(CORENLP_JAVA_LIBDIR, "lib", "Meteor-1.5.jar")


class Meteor:
    def __init__(self):
        self.meteor_cmd = [str(JAVA), "-jar", "-Xmx8G", METEOR_JAR, "-", "-", "-stdio", "-l", "en", "-norm"]

    def compute_score(self, gts, res):
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

        outs = outs.decode("utf-8")
        eval_input = "EVAL |||" + " ||| ".join(outs.strip().split("\n")) + "\n"

        proc = subprocess.Popen(
            self.meteor_cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        outs, _ = proc.communicate(input=eval_input.encode("utf-8"))
        proc.terminate()

        outs = outs.decode("utf-8")
        scores = [float(f) for f in outs.strip().split("\n")]

        return scores[-1], dict(zip(imgIds, scores[:-1]))

    def method(self):
        return "METEOR"


class MeteorBase:
    def __init__(self):
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

    def compute_score(self, gts, res):
        assert gts.keys() == res.keys()
        imgIds = gts.keys()
        scores = []

        eval_line = "EVAL"
        self.lock.acquire()
        for i in imgIds:
            assert len(res[i]) == 1
            stat = self._stat(res[i][0], gts[i])
            eval_line += " ||| {}".format(stat)

        self.meteor_p.stdin.write("{}\n".format(eval_line).encode())
        self.meteor_p.stdin.flush()
        for i in range(0, len(imgIds)):
            scores.append(float(self.meteor_p.stdout.readline().strip()))
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()

        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace("|||", "").replace("  ", " ")
        score_line = " ||| ".join(("SCORE", " ||| ".join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write("{}\n".format(score_line).encode())
        self.meteor_p.stdin.flush()
        return self.meteor_p.stdout.readline().decode().strip()

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace("|||", "").replace("  ", " ")
        score_line = " ||| ".join(("SCORE", " ||| ".join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write("{}\n".format(score_line))
        stats = self.meteor_p.stdout.readline().strip()
        eval_line = "EVAL ||| {}".format(stats)
        # EVAL ||| stats
        self.meteor_p.stdin.write("{}\n".format(eval_line))
        score = float(self.meteor_p.stdout.readline().strip())
        # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
        # thanks for Andrej for pointing this out
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()
        return score

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()
