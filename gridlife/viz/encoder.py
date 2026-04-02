import struct


def encode_palette_message(palette_bytes: bytes) -> bytes:
    """
    Encode palette for binary WebSocket transmission.
    Format: [0x01][768 bytes of RGB triplets]
    """
    return b"\x01" + palette_bytes


def encode_frame_message(grid_bytes: bytes, width: int, height: int) -> bytes:
    """
    Encode a raw uint8 grid frame for binary WebSocket transmission.
    Format: [0x02][uint16 width][uint16 height][width*height bytes]
    """
    header = struct.pack(">BHH", 0x02, width, height)
    return header + grid_bytes
