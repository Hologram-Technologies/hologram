@class:SP @id:SP-5 @spec:02-space-contract @phase:P3 @status:enforced
Feature: SP-5 — headless surface conformance
  Scenario: a headless space satisfies the Surface contract via the null projection
    Given a headless space's Surface
    When a workload is projected and an operator intent is submitted
    Then projection yields the canonical empty-projection κ and intent is refused with a typed headless error
