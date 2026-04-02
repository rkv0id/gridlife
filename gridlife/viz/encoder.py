import io

from PIL import Image


def encode_frame_jpeg(rgb_bytes: bytes, width: int, height: int, quality: int = 80) -> bytes:
    """Encode raw RGB bytes to JPEG."""
    img = Image.frombytes("RGB", (width, height), rgb_bytes)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
