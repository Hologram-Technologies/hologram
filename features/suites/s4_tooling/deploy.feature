@class:TL @id:TL-4 @spec:08-form-factor @phase:P5 @status:pending
Feature: TL-4 — deploy by κ
  Scenario: one app publishes to every rung by κ
    Given a compiled .holo app with a stable κ
    When hologram app publish stores it, announces it, and emits the boot page
    Then the same κ resolves and runs across every access rung with no cliff
