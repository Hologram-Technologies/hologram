@class:RM @id:RM-8 @spec:README @phase:P0 @status:enforced
Feature: host-language source frontend features
  Scenario: the frontend features are declared on the facade
    Given the frontend dependency snippet
    When I check the hologram facade manifest for frontends
    Then frontend-python, frontend-typescript, and frontend-rust are declared
