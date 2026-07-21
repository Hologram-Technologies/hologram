@class:RM @id:RM-18 @spec:README @phase:P0 @status:enforced
Feature: the Rust source frontend extracts builder graphs
  Scenario: the encoder graph is extracted and unrelated code ignored
    Given a Rust file with an encoder builder and unrelated code
    When the Rust frontend parses it
    Then the encoder graph is extracted and the unrelated code is ignored
