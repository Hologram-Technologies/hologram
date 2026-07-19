@class:MG @id:MG-3 @spec:06-migration @phase:P0.5 @status:pending
Feature: MG-3 — P0.5 de-risk spike
  Scenario: the de-risk spike proves composition before P1
    Given the async contract world and the sync compute hot path
    When the P0.5 vertical slice is built on native and wasm32
    Then composition is proven and the Send-bound question is resolved before any P1 move
