@class:MG @id:MG-2 @spec:06-migration @phase:P0 @status:enforced
Feature: P0 sync exit criteria
  Scenario: p0 exit criteria met
    Given holospaces pinned to its own repo
    When P0 synchronization completes
    Then holospaces ports to hologram HEAD, V&V is green, and the bridge tag is cut
