import hologram as hg
import unittest


class PackageTests(unittest.TestCase):
    def test_public_package_surface_imports(self):
        self.assertEqual(hg.f32, 8)
        self.assertIn("matmul", hg.op_names())
        self.assertIsNotNone(hg.Graph)


if __name__ == "__main__":
    unittest.main()
