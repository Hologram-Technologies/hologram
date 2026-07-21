@class:RM @id:RM-13 @spec:README @phase:P0 @status:enforced
Feature: one-call compile_from_source_language
  Scenario: a single-graph source compiles in one call
    Given a single-graph source in a host language
    When I call compile_from_source_language
    Then it returns a compiled archive
