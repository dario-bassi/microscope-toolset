import subprocess
import os
import signal
import sys
from .elasticsearch_db import ElasticSearchDB
import logging
import time

logger = logging.getLogger("NapariLauncher")
if not logger.handlers:
    handler = logging.FileHandler("napari_launch.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def _start_server(cmd):

    if sys.platform.startswith("win"):
        return subprocess.Popen(cmd, text=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)


def _stop_server(proc):
    if not proc:
        return

    if sys.platform.startswith("win"):
        subprocess.call(["taskkill", "/F", "/IM", "java.exe"])
        logger.info("Stopped Elasticsearch (killed java.exe)")
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass

def wait_for_es(max_wait=60, interval=1):
    es = ElasticSearchDB()
    waited = 0
    while waited < max_wait:
        if es.is_connected():
            logger.info("Elasticsearch is ready!")
            # close es client
            es.close()
            return True
        time.sleep(interval)
        waited += interval
        logger.info(f"Waiting for ES: {waited}/{max_wait}s")

    logger.error("Elasticsearch did not become ready in time")
    raise RuntimeError("Elasticsearch did not become ready in time")
