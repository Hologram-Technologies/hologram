@class:MG @id:MG-7 @spec:06-migration @phase:P3 @status:pending
Feature: MG-7 — holospaces V&V absorption
  Scenario: the holospaces CC catalog is absorbed into the unified conformance ledger
    Given holospaces' component-conformance (CC) catalog and its external authorities
    When its V&V is absorbed into hologram's conformance framework
    Then every CC witness runs under the one meta-gate against its external authority, never self-reference
