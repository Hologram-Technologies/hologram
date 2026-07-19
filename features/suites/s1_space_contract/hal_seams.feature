@class:SP @id:SP-4 @spec:02-space-contract @phase:P3 @status:enforced
Feature: SP-4 — deterministic HAL seams
  Scenario: the reference HAL seams are hermetic and deterministic
    Given the reference Entropy, Clock, and Spawner seams
    When entropy is drawn from two equally-seeded sources, the clock is advanced, and a background task is spawned
    Then the two entropy streams are identical, the clock reflects only explicit advances, and the spawned task is inert
