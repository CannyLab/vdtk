# Setup 3P packages to import correctly
import os
import sys

third_party_dir = os.path.dirname(os.path.abspath(__file__))

# TODO: This is a hack to get the 3P packages to import correctly. We should figure out a better way to do this.
sys.path.append(os.path.join(third_party_dir, "bleurt"))
sys.path.append(os.path.join(third_party_dir, "clip"))
