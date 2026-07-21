@class:RM @id:RM-27 @spec:README @phase:P6 @status:pending
Feature: the browser tab is the substrate
  Scenario: the wasm console signs in provisions and boots a userland
    Given the wasm32 holospaces Console bundle
    When it signs in, provisions a userland, and boots it
    Then the tab returns a κ snapshot without a server
