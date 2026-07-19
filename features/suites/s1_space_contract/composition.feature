@class:SP @id:SP-3 @spec:02-space-contract @phase:P0.5 @status:enforced
Feature: SP-3 — space composition
  Scenario: a space composes async network with sync storage and compute
    Given a Client over a space with a synchronous store and an async network seam
    When it drives compile then store then boot
    Then the workload runs end to end through the async-to-sync boundary
