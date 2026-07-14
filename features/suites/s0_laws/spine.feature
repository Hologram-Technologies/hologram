@class:LAW @id:LAW-1 @spec:00-overview @phase:P1 @status:pending
Feature: SPINE-1 — canonical bytes or nothing
  Scenario: canonical bytes or nothing
    Given a value with no canonical byte form
    When I attempt to construct a realization from it
    Then construction is unrepresentable and identity is only ever verified by re-derivation
