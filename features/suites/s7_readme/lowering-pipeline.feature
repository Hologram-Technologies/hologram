@class:RM @id:RM-11 @spec:README @phase:P0 @status:enforced
Feature: source frontends lower through one compile-time boundary
  Scenario: the documented lowering pipeline symbols compose
    Given native source text
    When I lower it through document, program, graph, and compiler
    Then each stage of the documented pipeline produces the next
