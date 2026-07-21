@class:RM @id:RM-35 @spec:README @phase:P0 @status:enforced
Feature: the just ci quality gate
  Scenario: the ci recipe chains fmt clippy test and supply-chain gate
    Given the repository Justfile
    When I read the ci recipe
    Then it chains fmt-check, clippy, test, and the supply-chain gate
