@class:SP @id:SP-1 @spec:02-space-contract @phase:P2 @status:pending
Feature: space contract conformance
  Scenario: passing the TCK is conformance
    Given a space implementing the hologram-space traits
    When it runs the hologram-tck battery
    Then passing the TCK is the definition of conformance
