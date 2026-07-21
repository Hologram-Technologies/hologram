@class:RM @id:RM-15 @spec:README @phase:P0 @status:enforced
Feature: compile a Python builder file through the CLI
  Scenario: the CLI compiles a selected Python graph
    Given a Python builder file on disk
    When I run the CLI compile verb with the frontend-python feature and a graph name
    Then it writes a compiled archive
