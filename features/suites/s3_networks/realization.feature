@class:NW @id:NW-1 @spec:04-networks @phase:P4 @status:pending
Feature: Network is a κ-realization
  Scenario: network embeds operand κs
    Given a Network built from a membership set and a policy
    When I call references() on its realization
    Then it yields the membership and policy operand κs with no side tables
