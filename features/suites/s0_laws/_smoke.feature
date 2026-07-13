@class:LAW @id:LAW-0 @spec:00-overview @phase:P0 @status:enforced
Feature: Conformance harness smoke
  Scenario: the harness discovers and runs feature files
    Given the conformance harness is wired
    Then it runs at least one scenario
