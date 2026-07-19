@class:TL @id:TL-1 @spec:05-tooling @phase:P5 @status:pending
Feature: exactly one binary
  Scenario: exactly one binary
    Given the built workspace
    When I list installed binaries
    Then exactly one is named hologram
