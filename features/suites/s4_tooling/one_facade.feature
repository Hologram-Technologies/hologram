@class:TL @id:TL-2 @spec:05-tooling @phase:P5 @status:pending
Feature: one public facade crate
  Scenario: one public crate
    Given a downstream consumer
    When it depends on the published crates
    Then it imports only the hologram facade with features, never a subcrate
