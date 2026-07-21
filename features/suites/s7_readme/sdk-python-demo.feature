@class:RM @id:RM-20 @spec:README @phase:P6 @status:pending
Feature: the Python SDK builds and runs a graph
  Scenario: the Python SDK compiles and executes a graph
    Given the hologram Python SDK
    When it builds, compiles, and executes a graph
    Then the session returns the output bytes
