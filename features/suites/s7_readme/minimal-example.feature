@class:RM @id:RM-10 @spec:README @phase:P0 @status:enforced
Feature: the README minimal example compiles and executes on the CPU backend
  Scenario: native source runs end to end
    Given the README's native source graph
    When I compile it to a .holo and load it on the CpuBackend
    Then executing against zero inputs yields one output buffer per port
