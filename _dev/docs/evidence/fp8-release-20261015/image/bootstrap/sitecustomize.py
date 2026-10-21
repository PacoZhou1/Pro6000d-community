"""Release bootstrap; an ordinary sitecustomize exception must not be ignored."""
import os
import sys
import traceback

try:
    import pro6000d_release
    pro6000d_release.activate()
except BaseException:
    traceback.print_exc(file=sys.stderr)
    sys.stderr.write("PRO6000D_RELEASE_FATAL: startup integrity/configuration failure; exit=78\n")
    sys.stderr.flush()
    os._exit(78)
