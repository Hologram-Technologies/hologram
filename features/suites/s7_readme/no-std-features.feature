@class:RM @id:RM-33 @spec:README @phase:P0 @status:partial
Feature: no_std facade feature composition
  Scenario: the no_std feature set resolves with default features off
    Given the no_std dependency snippet with default features off
    When I check the facade manifest for the no_std feature set
    Then backend, compiler, and exec compose without the std default
