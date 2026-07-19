@class:MG @id:MG-1 @spec:06-migration @phase:P1 @status:pending
Feature: always-green phase boundaries
  Scenario: each phase boundary is green
    Given the refactor phase sequence P0 through P6
    When a phase boundary is reached
    Then the full holospaces V&V passes before the next phase starts
