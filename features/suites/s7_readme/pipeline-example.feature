@class:RM @id:RM-9 @spec:README @phase:P0 @status:enforced
Feature: the end-to-end pipeline example
  Scenario: the pipeline example runs parse compile execute and address
    Given the hologram-cli pipeline example
    When I run it end to end
    Then it parses, compiles, executes, and addresses without error
