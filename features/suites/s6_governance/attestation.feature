@class:GV @id:GV-3 @spec:07-governance @phase:P5 @status:pending
Feature: R3 attestation
  Scenario: keys bind to κ-identity
    Given a space signing a session attestation
    When the signing key is published
    Then it is bound to a κ-addressed identity as content, never a second identity surface
