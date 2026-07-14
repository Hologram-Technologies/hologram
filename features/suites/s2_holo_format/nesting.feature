@class:HF @id:HF-2 @spec:03-holo-format @phase:P3 @status:pending
Feature: capability-attenuated app nesting
  Scenario: nested app cannot exceed parent
    Given a parent app with a CapabilitySet
    When it nests a child by κ ref with a delegated CapabilitySet
    Then the child's refs and capabilities are a subset of the parent's
