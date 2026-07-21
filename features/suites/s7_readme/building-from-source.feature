@class:RM @id:RM-34 @spec:README @phase:P0 @status:partial
Feature: building the workspace from source
  Scenario: the pipeline example runs from a source build
    Given a workspace source build
    When I run the pipeline example from source
    Then it completes the parse compile execute and address flow
