import numpy as np
from leaklens.labels import ascad_identity_label, TARGET_BYTE_INDEX


def test_target_byte_is_canonical():
    assert TARGET_BYTE_INDEX == 2, "Stage 1 reproduces against ANSSI canonical byte 2"


def test_label_known_pair():
    # AES SBox[0x53 ^ 0x00] = SBox[0x53] = 0xED
    pt  = np.zeros((1, 16), dtype=np.uint8); pt[0, 2] = 0x53
    key = np.zeros((1, 16), dtype=np.uint8)
    label = ascad_identity_label(pt, key)
    assert label[0] == 0xED
