@class:MG @id:MG-8 @spec:06-migration @phase:P3 @status:enforced
Feature: MG-8 — holospaces CS docs-conformance absorption
  Scenario: the holospaces CS catalog is absorbed into the unified conformance ledger
    Given holospaces' specification-conformance (CS) validators and their external standards
    When the docs V&V is absorbed into hologram's conformance framework
    Then every CS row runs V1-V8 against its external standard, never self-reference
