import os
import platform

# Enable multithreading on macOS High Sierra or higher
if platform.system() == "Darwin" and int(platform.mac_ver()[0].split(".")[1]) >= 12:
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
