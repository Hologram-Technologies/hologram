@class:MG @id:MG-6 @spec:06-migration @phase:P0 @status:pending
Feature: MG-6 — P0 license and review gate
  Scenario: license consent and restructuring review precede any move
    Given holospaces code contributed under MIT by a second contributor
    When P0 completes
    Then written dual-license consent and a restructuring spec review are recorded before any move
