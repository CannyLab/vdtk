#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

import os
import sys
import subprocess
import threading

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'


class Meteor:

    def __init__(self):
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, '-', '-', '-stdio', '-l', 'en', '-norm']

    def compute_score(self, gts, res):
        # assert (set(gts.keys()) == set(res.keys()))
        imgIds = list(res.keys())

        # Build the score evaluation line
        meteor_input = ''
        for i in imgIds:
            hypothesis_str, reference_list = (res[i][0], gts[i])
            hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
            score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
            meteor_input += '{}\n'.format(score_line)

        # Get the meteor output
        proc = subprocess.Popen(self.meteor_cmd,
                                cwd=os.path.dirname(os.path.abspath(__file__)),
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
        outs, _ = proc.communicate(input=meteor_input.encode('utf-8'))
        proc.terminate()

        outs = outs.decode('utf-8')
        eval_input = 'EVAL |||' + ' ||| '.join(outs.strip().split('\n')) + '\n'

        proc = subprocess.Popen(self.meteor_cmd,
                                cwd=os.path.dirname(os.path.abspath(__file__)),
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
        outs, _ = proc.communicate(input=eval_input.encode('utf-8'))
        proc.terminate()

        outs = outs.decode('utf-8')
        scores = [float(f) for f in outs.strip().split('\n')]

        return scores[-1], dict(zip(imgIds, scores[:-1]))

    def method(self):
        return "METEOR"
