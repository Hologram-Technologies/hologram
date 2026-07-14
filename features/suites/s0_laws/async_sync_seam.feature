@class:LAW @id:LAW-4 @spec:00-overview @phase:P2 @status:pending
Feature: LAW-4 — async contracts, sync compute
  Scenario: the session boundary is the only async-sync seam
    Given async I/O-shaped contract traits and a synchronous tensor hot path
    When a workload runs from storage through compute
    Then the only async-to-sync transition is the session boundary
