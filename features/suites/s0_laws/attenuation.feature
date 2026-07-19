@class:LAW @id:LAW-5 @spec:00-overview @phase:P2 @status:pending
Feature: capability attenuation only
  Scenario: delegation cannot amplify
    Given a capability set held by a grantor
    When the grantor delegates to a child
    Then the child's capabilities are a subset and amplification is unrepresentable
