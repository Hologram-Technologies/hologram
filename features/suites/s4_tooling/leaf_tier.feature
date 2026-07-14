@class:TL @id:TL-3 @spec:05-tooling @phase:P1 @status:pending
Feature: TL-3 — leaf tier dependency law
  Scenario: nothing depends on a leaf crate
    Given the tiers core, spaces, and leaf (facade plus Client, cli, packaging)
    When the workspace dependency graph is inspected
    Then dependencies flow core to spaces to leaf and no crate depends on a leaf crate
