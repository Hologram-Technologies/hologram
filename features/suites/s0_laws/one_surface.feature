@class:LAW @id:LAW-6 @spec:00-overview @phase:P3 @status:enforced
Feature: one programmatic surface
  Scenario: entry points are thin shells
    Given the CLI, FFI, and SDK entry points
    When I trace each to where behavior is defined
    Then every path resolves to the single Client facade
