@class:LAW @id:LAW-4 @spec:00-overview @phase:P2 @status:pending
Feature: LAW-4 — sync storage and compute, async network and lifecycle
  Scenario: the session boundary is the only async-sync seam
    Given synchronous storage and compute with async network and lifecycle
    When a workload runs from storage through compute
    Then the only async-to-sync transition is the network or boot boundary
