@class:RM @id:RM-17 @spec:README @phase:P0 @status:enforced
Feature: compile a TypeScript builder file through the CLI
  Scenario: the CLI compiles a selected TypeScript graph
    Given a TypeScript builder file on disk
    When I run the CLI compile verb with the frontend-typescript feature and a graph name
    Then it writes a compiled archive
