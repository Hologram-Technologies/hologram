@class:RM @id:RM-2 @spec:README @phase:P0 @status:partial
Feature: install alternatives — pin a version or build from source
  Scenario: the install script honors version and bin-dir overrides
    Given the repository's install.sh and the hologram-cli binary target
    When I inspect the installer's flags and the cargo install target
    Then version and bin-dir overrides are honored and the cli binary target exists
