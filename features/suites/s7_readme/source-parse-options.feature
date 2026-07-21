@class:RM @id:RM-12 @spec:README @phase:P0 @status:enforced
Feature: programmatic graph selection with SourceParseOptions
  Scenario: a named graph is selected and lowered
    Given a source document with a named graph
    When I select it with SourceParseOptions and lower it
    Then the named graph compiles
