@class:LAW @id:LAW-3 @spec:00-overview @phase:P2 @status:enforced
Feature: LAW-3 — contracts are hologram's, spaces are anyone's
  Scenario: the space contract is open to any repo
    Given the hologram-space contract with no sealed traits or crate-private seams
    When a space is implemented in an external repository
    Then it compiles against the published crates and is accepted with no in-tree privilege
