@class:RM @id:RM-25 @spec:README @phase:P6 @status:pending
Feature: run a holospace from Rust through the Platform Manager
  Scenario: provision open boot and suspend a holospace to a κ snapshot
    Given a signed-in Platform Manager over a peer
    When I provision a holospace, open it, boot it, and suspend it
    Then the suspend snapshot is a κ that resumes byte-identically
