@class:RM @id:RM-26 @spec:README @phase:P0 @status:partial
Feature: build a holospace container from the CLI
  Scenario: node put manifest and caps mint the container κs
    Given the container parts as bytes
    When I run node put, manifest, and caps
    Then each prints a κ-label and the container manifest addresses its parts
