@class:RM @id:RM-5 @spec:README @phase:P0 @status:enforced
Feature: the Client facade drives compile provision run over a space
  Scenario: a Client composes compile then provision then run
    Given a Client over the reference space
    When it drives compile then provision then run
    Then the workload produces its output through the one surface
