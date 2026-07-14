@class:MG @id:MG-4 @spec:06-migration @phase:P1 @status:pending
Feature: MG-4 — perf release gate
  Scenario: perf regression blocks a release
    Given roofline and kernel baselines captured at P1 preflight
    When a release re-runs hologram-bench and a kernel regresses past threshold
    Then the release is blocked, exactly as a κ break blocks it
