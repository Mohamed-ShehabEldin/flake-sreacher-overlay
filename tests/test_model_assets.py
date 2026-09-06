import hashlib
import json
import unittest

from flake_searcher.paths import ASSETS_ROOT, MODELS_ROOT


EXPECTED_MODELS = {
    "TIT_10x.h5",
    "WSe2_EVE Microscope_10x_ALPHA.h5",
    "WSe2_EVE Microscope_20x_ALPHA.h5",
    "WSe2_EVE Microscope_50x - BETA_M1(90+).h5",
    "WSe2_EVE Microscope_50x - BETA_M2(95+).h5",
    "WSe2_EVE Microscope_50x_ALPHA.h5",
}


class ModelAssetTests(unittest.TestCase):
    def test_expected_models_are_present(self):
        self.assertEqual({path.name for path in MODELS_ROOT.glob("*.h5")}, EXPECTED_MODELS)

    def test_model_bytes_match_asset_manifest(self):
        manifest = json.loads((ASSETS_ROOT / "manifest.json").read_text(encoding="utf-8"))
        expected_hashes = manifest["detector_models"]
        self.assertEqual(set(expected_hashes), EXPECTED_MODELS)
        for filename, expected_hash in expected_hashes.items():
            with self.subTest(model=filename):
                actual_hash = hashlib.sha256((MODELS_ROOT / filename).read_bytes()).hexdigest()
                self.assertEqual(actual_hash, expected_hash)

    def test_all_models_load_without_compilation(self):
        try:
            from tensorflow import keras
        except ImportError:
            self.skipTest("TensorFlow is not installed in this test environment")

        for path in sorted(MODELS_ROOT.glob("*.h5")):
            with self.subTest(model=path.name):
                model = keras.models.load_model(str(path), compile=False)
                self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
