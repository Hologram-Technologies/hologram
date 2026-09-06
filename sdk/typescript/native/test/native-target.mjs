import assert from "node:assert/strict";
import { nativeTargetTag } from "../scripts/native-target.mjs";

assert.equal(
  nativeTargetTag({ platform: "linux", arch: "x64", glibcVersionRuntime: "2.39" }),
  "linux-x64",
);
assert.equal(
  nativeTargetTag({ platform: "linux", arch: "arm64", glibcVersionRuntime: null }),
  "linux-arm64-musl",
);
assert.equal(nativeTargetTag({ platform: "darwin", arch: "arm64" }), "darwin-arm64");
assert.equal(nativeTargetTag({ platform: "win32", arch: "x64" }), "win32-x64");
