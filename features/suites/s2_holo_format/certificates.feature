@class:HF @id:HF-3 @spec:03-holo-format @phase:P4 @status:enforced
Feature: per-layer certificates
  Scenario: per-layer certificates verify
    Given a .holo v3 with per-layer certificates
    When I inspect it through the Client surface
    Then every certificate verifies and none is stripped
