// Keep build-time and smoke-test artifact selection identical to the runtime loader in
// src/index.ts. Linux musl needs a distinct tag because glibc and musl addons are not
// interchangeable.
export function nativeTargetTag({
  platform = process.platform,
  arch = process.arch,
  glibcVersionRuntime = process.report?.getReport?.()?.header?.glibcVersionRuntime,
} = {}) {
  if (platform === "linux" && !glibcVersionRuntime) {
    return `linux-${arch}-musl`;
  }
  return `${platform}-${arch}`;
}
