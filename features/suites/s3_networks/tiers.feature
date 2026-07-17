@class:NW @id:NW-2 @spec:04-networks @phase:P5 @status:enforced
Feature: network tiers gate capability
  Scenario: tiers gate at the boundary
    Given public, restricted, and private network tiers
    When a peer attempts store/fetch/announce
    Then the capability check happens at the protocol boundary, not in business logic
