"""
scripts/colab_notebook_setup.py

NOT meant to be run as `python -m scripts.colab_notebook_setup` — this
is reference code to COPY-PASTE into the first cell of your Colab
notebook. It's kept as a .py file here just so it's version-controlled
alongside the rest of the project instead of living only inside a
notebook (which is harder to diff/review on GitHub).

===================== PASTE INTO COLAB CELL 1 =====================

# Clone your repo (or upload the project zip and unzip it instead)
!git clone https://github.com/YOUR_ORG/YOUR_REPO.git
%cd YOUR_REPO

!pip install -r requirements.txt -q

import sys
sys.path.append('.')

from src.utils.colab_utils import mount_drive, colab_output_root, print_colab_session_tips

mount_drive()  # will prompt you to authorize Drive access — click through it

from config import Config
config = Config()
config.output_root = colab_output_root()  # <- THIS is the key line:
                                            #    points outputs/checkpoints/logs
                                            #    at Drive instead of ephemeral local disk

# also worth pointing your DATASET at Drive, so you don't have to
# re-upload images every session:
# config.data_root = Path("/content/drive/MyDrive/your_dataset_folder")

print_colab_session_tips()

===================== PASTE INTO COLAB CELL 2 (repeat after any disconnect) =====================

!python -m scripts.run_all_experiments --backbones resnet50

# (swap resnet50 for whichever architecture this session is responsible
#  for, or omit --backbones to run everything configured in config.py)

=====================================================================

After a disconnect: reconnect the runtime, re-run CELL 1 (Drive
remounts, doesn't re-download anything), then re-run CELL 2 with the
SAME --backbones argument — completed runs are skipped automatically,
the in-progress run resumes from its last checkpointed epoch.
"""
