@class:RM @id:RM-28 @spec:README @phase:P0 @status:enforced
Feature: composing a minimal Space for the Client
  Scenario: a minimal reference space is accepted by the Client
    Given a minimal Space composed from the reference pieces
    When I build a Client over it
    Then the Client accepts the space and reaches a contract-mediated operation
