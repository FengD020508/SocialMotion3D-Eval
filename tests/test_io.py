from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from socialmotion3d_eval.io import load_obd_speed_kmh


class IoTests(unittest.TestCase):
    def test_obd_prefers_source_id_for_extracted_clips(self) -> None:
        xml = """<root>
          <frame id="0" source_id="20826" OBD_speed="18.0" />
          <frame id="1" source_id="20827" OBD_speed="36.0" />
        </root>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "obd.xml"
            path.write_text(xml, encoding="utf-8")
            result = load_obd_speed_kmh(path)
        self.assertEqual(result, {20826: 18.0, 20827: 36.0})


if __name__ == "__main__":
    unittest.main()
