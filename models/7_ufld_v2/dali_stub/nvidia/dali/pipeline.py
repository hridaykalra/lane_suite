class Pipeline:
    """Stub — NVIDIA DALI is only used by this repo's TRAINING dataloader
    (data/dali_data.py), never by inference. LaneSuite only runs inference,
    so this stub exists purely to satisfy an unused import at module-load
    time, without requiring the real (large, platform-specific) DALI
    library to be installed. See the LaneSuite README for details."""
    pass
