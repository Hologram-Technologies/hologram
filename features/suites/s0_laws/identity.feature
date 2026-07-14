@class:LAW @id:LAW-2 @spec:00-overview @phase:P1 @status:pending
Feature: κ-only identity
  Scenario: no second naming surface
    Given a contract type and a stored realization
    When I enumerate every identity-bearing field
    Then none is a UUID, PeerId, Multiaddr, path, or hostname, and no transport id leaks
