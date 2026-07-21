@class:RM @id:RM-4 @spec:README @phase:P0 @status:enforced
Feature: quickstart library features
  Scenario: the documented tensor-engine features are declared on the facade
    Given the quickstart dependency snippet's feature list
    When I check the hologram facade manifest
    Then archive, backend, compiler, and exec are declared features
