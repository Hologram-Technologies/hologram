@class:SP @id:SP-3 @spec:02-space-contract @phase:P0.5 @status:pending
Feature: SP-3 — space composition
  Scenario: a space composes async storage and sync compute
    Given a Space providing async storage and the synchronous compute hot path
    When Client drives compile then open then boot on a native and a wasm target
    Then the slice runs on both targets and the Send-bound policy is settled by evidence
