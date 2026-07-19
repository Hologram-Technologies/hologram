@class:HF @id:HF-1 @spec:03-holo-format @phase:P4 @status:enforced
Feature: .holo v3 is the one container
  Scenario: single format covers tensor-only
    Given a tensor-only archive
    When I open it as a .holo v3 application
    Then it is the degenerate single-layer case of the one format
