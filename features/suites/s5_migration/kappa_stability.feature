@class:MG @id:MG-5 @spec:06-migration @phase:P1 @status:enforced
Feature: MG-5 — κ-stability across moves
  Scenario: golden vectors re-derive bit-identically across moves
    Given frozen golden vectors for the σ-axis and the realization canonical forms
    When each is re-derived from the same inputs
    Then every vector yields its frozen κ, bit-for-bit
