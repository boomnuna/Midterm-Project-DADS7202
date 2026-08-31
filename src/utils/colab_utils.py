"""
src/utils/colab_utils.py

Google Colab has two failure modes that matter here:
  1. Local disk (/content/...) is EPHEMERAL — wiped completely when the
     runtime disconnects/recycles. Anything not saved to Google Drive
     (or downloaded manually) is just gone.
  2. Sessions disconnect without warning — free tier has an idle
     timeout (~90 min with no interaction) AND a hard max session
     length (~12 hrs), and the GPU can be reclaimed at any time under
     heavy platform load.

The fix for (1) is: mount Drive, point config.output_root at it.
The fix for (2) is: Trainer/ExperimentRunner's checkpoint+resume logic
(see trainer.py, repeated_runs.py) — just re-run your script after a
disconnect and it picks up where it left off, AS LONG AS outputs/ is on
Drive so it survived the disconnect.
"""

from pathlib import Path


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive(mount_point: str = "/content/drive") -> str | None:
    """
    Call this ONCE at the very top of your Colab notebook, before
    building Config(). Will prompt you to authorize access the first
    time (a link + confirmation, standard Colab flow).
    """
    if not is_colab():
        print("Not running in Colab — skipping Drive mount (no-op).")
        return None

    from google.colab import drive
    drive.mount(mount_point)
    print(f"Drive mounted at {mount_point}")
    return mount_point


def colab_output_root(drive_subfolder: str = "MyDrive/dl_hw_project_outputs") -> str:
    """
    Returns the path to use as config.output_root.
      - in Colab: a folder INSIDE the mounted Drive (survives disconnects)
      - outside Colab: falls back to a local "outputs" folder (so this
        function is safe to call unconditionally, e.g. if you're testing
        the same notebook code locally)
    """
    if is_colab():
        path = f"/content/drive/{drive_subfolder}"
        Path(path).mkdir(parents=True, exist_ok=True)
        return path
    return "outputs"


def print_colab_session_tips():
    """
    Call this once after setup, just to print a checklist on-screen at
    the start of a notebook session — cheap insurance against forgetting
    a step after a disconnect at 2am.
    """
    print("""
COLAB SESSION CHECKLIST
========================
1. Drive mounted?              -> mount_drive() should have printed "Drive mounted"
2. config.output_root on Drive? -> should NOT start with /content/ alone
                                    (that's local, ephemeral disk)
3. Runtime type = GPU?          -> Runtime menu > Change runtime type > GPU
4. If disconnected: just re-run your training cell/script again.
   Completed (backbone, seed) runs are skipped automatically; an
   in-progress run resumes from its last saved epoch, not from scratch.
5. Long unattended runs: Colab can disconnect idle sessions even with
   a tab open. Keeping the browser tab focused/active reduces this, but
   nothing guarantees a session survives many hours unattended — that's
   exactly why the checkpoint/resume system exists, so don't rely on
   preventing disconnects, rely on recovering from them cheaply.
""")
