@class:GV @id:GV-4 @spec:07-governance @phase:P6 @status:enforced
Feature: R4 data governance
  Scenario: capability checks at the boundary
    Given a network capability policy with quotas
    When a peer stores, fetches, or announces content
    Then the capability check is at the import/protocol boundary and accounting is per-capability
