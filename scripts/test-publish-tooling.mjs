import assert from "node:assert/strict";
import { sha512Integrity, requireMatchingIntegrity, singlePackRow } from "./publish-npm.mjs";

const expected = "sha512-Z3JlZW4=";
assert.equal(
  sha512Integrity(Buffer.from("green")),
  "sha512-0hYa4xk4ee3Zyovrt1tAVvJ1pNrPpPQsR3jQY+EUnz4OQXbkLu/OVz0adD6Jz16EjqCdWTs61P5B5eeeqXBHMw==",
);
assert.doesNotThrow(() => requireMatchingIntegrity("pkg", "1.0.0", expected, null));
assert.doesNotThrow(() => requireMatchingIntegrity("pkg", "1.0.0", expected, expected));
assert.throws(
  () => requireMatchingIntegrity("pkg", "1.0.0", expected, "sha512-different"),
  /public integrity .* differs from packed/,
);
assert.equal(singlePackRow('[{"filename":"legacy.tgz"}]', "legacy").filename, "legacy.tgz");
assert.equal(
  singlePackRow('{"@tryhologram/sdk":{"filename":"npm12.tgz"}}', "npm12").filename,
  "npm12.tgz",
);
assert.throws(() => singlePackRow("{}", "empty"), /unexpected manifest/);
assert.throws(
  () => singlePackRow('[{"filename":"one.tgz"},{"filename":"two.tgz"}]', "many"),
  /unexpected manifest/,
);

console.log("PASS test-publish-tooling");
