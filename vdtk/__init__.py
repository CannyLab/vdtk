import os
import platform

# Enable multithreading on macOS High Sierra or higher
if platform.system() == "Darwin" and int(platform.mac_ver()[0].split(".")[1]) >= 12:
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Tensorflow logging is annoying
os.environ["USE_TORCH"] = "True"  # Transformers doesn't like tensorflow
os.environ["TOKENIZERS_PARALLELISM"] = "False"  # Parallelism doesn't work for hugginface tokenizers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Transformers logging is annoying
os.environ["DISABLE_TQDM"] = "True"  # Disable TQDM when computing mauve scores

__version__ = "0.3.0"
