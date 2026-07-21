@class:RM @id:RM-24 @spec:README @phase:P0 @status:enforced
Feature: content-address and compose model parts as UOR-ADDR κ-labels
  Scenario: two κ addresses compose order-independently
    Given two model-part rings addressed to κ-labels
    When I compose them in both orders
    Then compose_model yields the same model identity regardless of order
