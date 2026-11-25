"""Optional dependency helpers retained for compatibility with the uploaded source."""
from importlib.util import find_spec

def is_bnb_available() -> bool:
    return find_spec("bitsandbytes") is not None
