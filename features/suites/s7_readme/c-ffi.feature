@class:RM @id:RM-32 @spec:README @phase:P0 @status:enforced
Feature: the C ABI drives the compile and session pipeline
  Scenario: the C ABI compiles loads executes and closes a session
    Given native source and the C ABI entry points
    When I compile the source, load a session, execute it, and close it
    Then the session handle drives the full pipeline and the ABI version is reported
