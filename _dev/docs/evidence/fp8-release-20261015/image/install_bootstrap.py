"""Build-time CPU installer. No vLLM import, CUDA initialization or compilation."""
import importlib.util
import json
import sys
import sysconfig
from pathlib import Path

ROOT = Path("/opt/pro6000d")
spec = importlib.util.spec_from_file_location("pro6000d_install_audit", ROOT / "bootstrap/pro6000d_release.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
manifest = module.audit(ROOT, module.installed_vllm_root(), module.BASE_SITE)
bootstrap = ROOT / "bootstrap"
assert (bootstrap / "sitecustomize.py").is_file()
purelib = Path(sysconfig.get_path("purelib"))
assert purelib.is_dir()
pth = purelib / "000_pro6000d_release.pth"
assert not pth.exists(), "release .pth already installed"
line = "import sys; sys.path.insert(0, '/opt/pro6000d/bootstrap')\n"
pth.write_text(line)
print(json.dumps({"installed": str(pth), "python": sys.version,
                  "manifest_sha256": module.sha(ROOT / "manifest.json")}, sort_keys=True))
