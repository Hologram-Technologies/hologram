@class:RM @id:RM-16 @spec:README @phase:P0 @status:enforced
Feature: the TypeScript source frontend extracts builder graphs
  Scenario: the encoder graph is extracted and unrelated code ignored
    Given a TypeScript file with an encoder builder and unrelated code
    When the TypeScript frontend parses it
    Then the encoder graph is extracted and the unrelated code is ignored
