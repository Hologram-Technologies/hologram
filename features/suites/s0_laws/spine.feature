@class:LAW @id:LAW-1 @spec:00-overview @phase:P1 @status:enforced
Feature: SPINE-1 — canonical bytes or nothing
  Scenario: canonical bytes or nothing
    Given a realization addressed only by its canonical bytes
    When its identity is checked
    Then re-derivation of the canonical bytes verifies, and any tampering is rejected
