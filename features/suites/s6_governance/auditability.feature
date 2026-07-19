@class:GV @id:GV-2 @spec:07-governance @phase:P6 @status:enforced
Feature: R2 auditability
  Scenario: one audit seam, no bypass
    Given lifecycle transitions spawn, suspend, resume, terminate
    When each transition occurs
    Then it emits through one seam that can be pointed at the κ-chain and no path bypasses it
