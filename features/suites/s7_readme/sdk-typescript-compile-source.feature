@class:RM @id:RM-23 @spec:README @phase:P6 @status:pending
Feature: the TypeScript SDK compiles native source
  Scenario: compileSource compiles native source text
    Given native source text and the TypeScript SDK
    When I call compileSource
    Then it returns a compiled archive
