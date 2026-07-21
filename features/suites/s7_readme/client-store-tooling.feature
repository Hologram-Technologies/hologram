@class:RM @id:RM-29 @spec:README @phase:P0 @status:enforced
Feature: store, GC, and app tooling on one Client handle
  Scenario: the Client exposes get pin ls inspect thin and open
    Given a Client with a compiled and provisioned workload
    When I exercise get, pin, ls, inspect, thin, and open on the one handle
    Then each store and app-tooling operation succeeds on the same surface
