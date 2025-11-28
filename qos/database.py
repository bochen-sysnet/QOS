"""
Compatibility shim to keep `import qos.database` working; re-export the backend DB.
"""
from qos.backends.database import *  # noqa: F401,F403
