@class:RM @id:RM-19 @spec:README @phase:P0 @status:enforced
Feature: compile a Rust builder file through the CLI
  Scenario: the CLI compiles a selected Rust graph
    Given a Rust builder file on disk
    When I run the CLI compile verb with the frontend-rust feature and a graph name
    Then it writes a compiled archive
