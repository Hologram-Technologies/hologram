@class:RM @id:RM-6 @spec:README @phase:P0 @status:enforced
Feature: using the tensor engine — the library feature set
  Scenario: the tensor-engine feature set resolves on the facade
    Given the tensor-engine dependency snippet
    When I resolve its features on the hologram facade
    Then the archive, backend, compiler, and exec features resolve
