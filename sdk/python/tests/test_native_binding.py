import os
import struct
import tempfile
import unittest

import hologram as hg
import hologram._hologram as native

F32_ONE_BLAKE3 = "1628d491647767ca75acfd2183da4631eef590e9c8129503574e36927c16521c"


def require_native():
    try:
        native.assert_compatible()
    except native.NativeError as exc:
        if exc.code == 0:
            raise unittest.SkipTest(str(exc)) from exc
        raise


class NativeBindingTests(unittest.TestCase):
    def setUp(self):
        require_native()

    def test_native_binding_compiles_graph(self):
        graph = hg.Graph("native_smoke", native=native)
        x = graph.input("x", shape=[1])
        y = x.relu(shape=[1])
        archive = graph.output("y", y).compile()

        self.assertEqual(archive[:4], b"HOLO")

    def test_native_binding_compiles_source_string(self):
        archive = hg.compile_source("input x :1\nop relu x :1 as=y\noutput y\n")

        with hg.Session.load(archive) as session:
            outputs = session.execute({"x": struct.pack("<f", -2.0)})

        self.assertEqual(struct.unpack("<f", outputs["y"])[0], 0.0)

    def test_native_binding_compiles_txt_source_file(self):
        fd, path = tempfile.mkstemp(prefix="hologram-python-source-", suffix=".txt")
        try:
            os.write(fd, b"input x :1\nop relu x :1 as=y\noutput y\n")
        finally:
            os.close(fd)
        try:
            archive = hg.compile_source_file(path)
        finally:
            os.remove(path)

        self.assertEqual(archive[:4], b"HOLO")

    def test_native_binding_rejects_unlowered_attrs(self):
        builder = native.source_builder()
        try:
            with self.assertRaises(hg.BadAttrError) as raised:
                builder.op("gemm", [], as_="y", alpha=0.5)
            self.assertEqual(raised.exception.code, 4)
        finally:
            builder.free()

    def test_native_binding_rejects_bad_const_ref_hash(self):
        builder = native.source_builder()
        try:
            with self.assertRaises(hg.ExternalTensorError) as raised:
                builder.const_ref("w", shape=[1], file="weights.bin", blake3="not-hex")
            self.assertEqual(raised.exception.code, 6)
        finally:
            builder.free()

    def test_native_session_rejects_bad_archive(self):
        with self.assertRaises(hg.ArchiveLoadError) as raised:
            hg.Session.load(b"not-a-holo")
        self.assertEqual(raised.exception.code, 7)

    def test_native_session_loads_executes_and_introspects(self):
        archive = _relu_archive()

        with hg.Session.load(archive) as session:
            self.assertEqual(session.input_count, 1)
            self.assertEqual(session.output_count, 1)
            self.assertGreater(session.kernel_count, 0)
            self.assertEqual(session.input_name(0), "x")
            self.assertEqual(session.output_name(0), "y")
            self.assertEqual(session.input_shape(0), (1,))
            self.assertIsInstance(session.output_shape(0), tuple)
            self.assertEqual(session.output_byte_len(0), 4)
            self.assertEqual(session.input_dtype(0), hg.f32)
            self.assertEqual(session.output_dtype(0), hg.f32)
            self.assertIsNone(session.extension("missing"))
            self.assertEqual(len(session.archive_fingerprint), 32)

            outputs = session.execute({"x": struct.pack("<f", -2.0)})

        self.assertEqual(set(outputs), {"y"})
        self.assertEqual(struct.unpack("<f", outputs["y"])[0], 0.0)

    def test_native_session_accepts_single_bytes_input(self):
        with hg.Session.load(_relu_archive()) as session:
            outputs = session.execute(struct.pack("<f", -1.0))
        self.assertEqual(struct.unpack("<f", outputs["y"])[0], 0.0)

    def test_native_session_rejects_missing_named_input(self):
        with hg.Session.load(_relu_archive()) as session:
            with self.assertRaises(hg.InvalidArgumentError) as raised:
                session.execute({})
            self.assertEqual(raised.exception.code, 10)

    def test_native_session_rejects_bad_input_bytes(self):
        with hg.Session.load(_relu_archive()) as session:
            with self.assertRaises(hg.ExecutionError) as raised:
                session.execute(b"")
            self.assertEqual(raised.exception.code, 8)

    def test_native_const_ref_compiles_and_embeds_bytes(self):
        data = struct.pack("<f", 1.0)
        fd, path = tempfile.mkstemp(prefix="hologram-python-const-ref-", suffix=".bin")
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        try:
            graph = hg.Graph("native_const_ref", native=native)
            x = graph.input("x", shape=[1])
            w = graph.const_ref("w", shape=[1], file=path, blake3=F32_ONE_BLAKE3, byte_len=4)
            archive = graph.output("y", x.add(w, shape=[1])).compile()
        finally:
            os.remove(path)

        with hg.Session.load(archive) as session:
            outputs = session.execute({"x": struct.pack("<f", 0.0)})
        self.assertEqual(struct.unpack("<f", outputs["y"])[0], 1.0)


def _relu_archive() -> bytes:
    graph = hg.Graph("native_session_smoke", native=native)
    x = graph.input("x", shape=[1])
    return graph.output("y", x.relu(shape=[1])).compile()


if __name__ == "__main__":
    unittest.main()
