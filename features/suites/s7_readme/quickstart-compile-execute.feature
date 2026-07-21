@class:RM @id:RM-3 @spec:README @phase:P0 @status:enforced
Feature: quickstart — compile native source then execute
  Scenario: the CLI compiles a source graph and executes the archive
    Given a native hologram source file
    When I run the CLI compile then execute verbs on it
    Then the archive round-trips and reports one length per output port
