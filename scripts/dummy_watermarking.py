"""Dummy watermarking module to replace Sesame's watermarking when silentcipher is not installed."""
CSM_1B_GH_WATERMARK = b'\x00' * 16

def load_watermarker(device=None):
    """Return a dummy watermarker that does nothing."""
    class DummyWatermarker:
        pass
    return DummyWatermarker()

def watermark(watermarker, audio, sample_rate, watermark_data):
    """Return audio unchanged (no watermarking)."""
    return audio, sample_rate
