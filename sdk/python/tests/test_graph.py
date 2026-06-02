from hologram import BadAttrError, Graph, f32, op_names
import unittest


class Recorder:
    def __init__(self):
        self.events = []

    def feature_supported(self, feature):
        return True

    def source_builder(self):
        return self

    def input(self, name, *, dtype, shape=None):
        self.events.append(("input", name, dtype, shape))
        return name

    def const(self, name, *, dtype, shape, values):
        self.events.append(("const", name, dtype, shape, values))
        return name

    def const_ref(self, name, *, dtype, shape, file, blake3, byte_len=None, byte_offset=0):
        self.events.append(("const_ref", name, dtype, shape, file, blake3, byte_len, byte_offset))
        return name

    def op(self, op, inputs, *, as_, **attrs):
        self.events.append(("op", as_, op, tuple(inputs), attrs))
        return as_

    def output(self, name, *, source=None):
        self.events.append(("output", name, source))

    def compile(self):
        return b"archive"


def golden_external_ref_events():
    return [
        ("input", "x", f32, (2, 3)),
        ("const_ref", "w", f32, (3, 2), "weights.bin", "0" * 64, 24, 0),
        ("op", "_t0", "matmul", ("x", "w"), {"shape": [2, 2]}),
        ("output", "y", "_t0"),
    ]


class GraphTests(unittest.TestCase):
    def test_chainable_graph_emits_builder_calls(self):
        native = Recorder()
        g = Graph("encoder", native=native)
        x = g.input("x", dtype=f32, shape=[2, 3])
        w = g.const_ref("w", shape=[3, 2], file="weights.bin", blake3="0" * 64)
        y = x.matmul(w, shape=[2, 2]).relu()

        self.assertEqual(g.output("y", y).compile(), b"archive")
        self.assertIn(("op", "_t0", "matmul", ("x", "w"), {"shape": [2, 2]}), native.events)
        self.assertIn(("op", "_t1", "relu", ("_t0",), {}), native.events)
        self.assertIn(("output", "y", "_t1"), native.events)

    def test_low_level_escape_hatch_keeps_explicit_alias(self):
        native = Recorder()
        g = Graph(native=native)
        g.input("x")
        g.const("w", shape=[1], values=[1.0])
        g.op("add", ["x", "w"], as_="y")
        g.output("y").compile()

        self.assertIn(("op", "y", "add", ("x", "w"), {}), native.events)
        self.assertIn(("output", "y", "y"), native.events)

    def test_rejects_attrs_not_supported_by_op_metadata(self):
        g = Graph()
        x = g.input("x")

        with self.assertRaises(BadAttrError) as raised:
            x.relu(axis=1)
        self.assertIn("relu: unsupported attrs: axis", str(raised.exception))

    def test_generated_op_names_are_visible(self):
        self.assertIn("matmul", op_names())

    def test_external_ref_golden_emits_parser_equivalent_builder_contract(self):
        native = Recorder()
        g = Graph("encoder", native=native)
        x = g.input("x", dtype=f32, shape=[2, 3])
        w = g.const_ref("w", dtype=f32, shape=[3, 2], file="weights.bin", blake3="0" * 64, byte_len=24)
        y = x.matmul(w, shape=[2, 2])

        self.assertEqual(g.output("y", y).compile(), b"archive")
        self.assertEqual(native.events, golden_external_ref_events())


if __name__ == "__main__":
    unittest.main()
