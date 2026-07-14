@class:GV @id:GV-1 @spec:07-governance @phase:P5 @status:enforced
Feature: R1 traceability by κ
  Scenario: references yields full provenance
    Given a new realization built from known operand κs
    When I call references() on it
    Then the returned set equals the full provenance closure with no side tables
