@class:RM @id:RM-22 @spec:README @phase:P6 @status:pending
Feature: the Python SDK compiles native source files
  Scenario: compile_source_file compiles a native source graph
    Given a native source file and the Python SDK
    When I call compile_source_file
    Then it returns a compiled archive
