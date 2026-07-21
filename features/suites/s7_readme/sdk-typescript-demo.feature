@class:RM @id:RM-21 @spec:README @phase:P6 @status:pending
Feature: the TypeScript SDK builds and runs a graph
  Scenario: the TypeScript SDK compiles and executes a graph
    Given the tryhologram TypeScript SDK and native binding
    When it builds, compiles, and executes a graph
    Then the session returns the output bytes
