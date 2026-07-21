@class:RM @id:RM-30 @spec:README @phase:P0 @status:enforced
Feature: the hologram CLI tensor verbs
  Scenario: compile inspect execute and bench operate on one archive
    Given a compiled .holo archive
    When I run the inspect, execute, and bench verbs on it
    Then each verb reports on the archive without error
