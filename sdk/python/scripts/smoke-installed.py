import struct

import hologram as hg


def main() -> None:
    assert hg.f32 == 8
    assert "matmul" in hg.op_names()
    graph = hg.Graph("installed_smoke")
    x = graph.input("x", shape=[1])
    archive = graph.output("y", x.relu(shape=[1])).compile()
    assert archive[:4] == b"HOLO"
    with hg.Session.load(archive) as session:
        outputs = session.execute({"x": struct.pack("<f", -1.0)})
    assert struct.unpack("<f", outputs["y"])[0] == 0.0


if __name__ == "__main__":
    main()
