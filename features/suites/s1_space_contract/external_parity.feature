@class:SP @id:SP-2 @spec:02-space-contract @phase:P4 @status:pending
Feature: external-repo space parity
  Scenario: external space is first-class
    Given a space living in an external repository depending only on published crates
    When it runs the TCK as a dev-dependency
    Then Client accepts it with no facade change
