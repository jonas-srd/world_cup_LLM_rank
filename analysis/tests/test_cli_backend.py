from __future__ import annotations

import subprocess
import sys


def test_cli_import_does_not_import_matplotlib_pyplot(project_root):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import soccerarena_analysis.cli; "
                "raise SystemExit('matplotlib.pyplot' in sys.modules)"
            ),
        ],
        cwd=project_root,
        check=False,
    )
    assert result.returncode == 0
