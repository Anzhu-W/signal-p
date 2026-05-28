"""Canonical ASCAD label: identity model on SBox output, byte index 2."""
import numpy as np
from leaklens.aes import SBOX

TARGET_BYTE_INDEX = 2  # ANSSI default. DO NOT CHANGE WITHOUT UPDATING README.


def ascad_identity_label(plaintext: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Returns SBox[plaintext[:,2] XOR key[:,2]] as int64."""
    return SBOX[plaintext[:, TARGET_BYTE_INDEX] ^ key[:, TARGET_BYTE_INDEX]].astype(np.int64)
