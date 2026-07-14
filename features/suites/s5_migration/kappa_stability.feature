@class:MG @id:MG-5 @spec:06-migration @phase:P1 @status:pending
Feature: MG-5 — κ-stability across moves
  Scenario: golden vectors re-derive bit-identically across moves
    Given golden vectors for every realization kind, a v2 .holo, and a SPINE-4 frame
    When a crate is renamed or relocated
    Then every golden vector re-derives to bit-identical bytes and the same κ
