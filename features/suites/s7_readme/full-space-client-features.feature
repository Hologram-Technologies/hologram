@class:RM @id:RM-7 @spec:README @phase:P0 @status:enforced
Feature: the full, space, and client features
  Scenario: full enables the tensor-engine modules and space plus client are available
    Given the facade feature manifest
    When I resolve the full, space, and client features
    Then full enables the tensor-engine modules and space and client are declared
