@class:RM @id:RM-14 @spec:README @phase:P0 @status:enforced
Feature: the Python source frontend extracts builder graphs
  Scenario: the encoder graph is extracted and unrelated code ignored
    Given a Python file with an encoder builder and unrelated code
    When the Python frontend parses it
    Then the encoder graph is extracted and the unrelated code is ignored
