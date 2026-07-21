@class:RM @id:RM-1 @spec:README @phase:P0 @status:partial
Feature: the install script is a portable one-liner
  Scenario: install.sh downloads a platform binary into the local bin dir
    Given the repository's install.sh
    When I read its installer contract
    Then it is POSIX sh that installs a prebuilt binary into the local bin dir
