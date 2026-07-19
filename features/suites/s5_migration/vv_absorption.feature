@class:MG @id:MG-7 @spec:06-migration @phase:P3 @status:enforced
Feature: MG-7 — holospaces V&V absorption
  Scenario: the holospaces CC catalog is absorbed into the unified conformance ledger
    Given the holospaces CC catalog in the unified conformance ledger
    When the CC bijection audit binds every row to its witness test
    Then every CC row binds to a present, named witness — none by self-reference
