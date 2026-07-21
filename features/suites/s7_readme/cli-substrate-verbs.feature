@class:RM @id:RM-31 @spec:README @phase:P0 @status:partial
Feature: the hologram CLI substrate verbs
  Scenario: node app and network verbs operate over a store
    Given a node store and content bytes
    When I put content and then get and verify it by κ
    Then the bytes round-trip and re-derive to the same κ
