#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: CommandLineTool
$namespaces:
  sf: "https://streamflow.org/cwl#"
baseCommand: ["singularity", "exec"]
arguments:
  - position: 2
    vaueFrom: "python"
  - position: 3
    valueFrom: "/opt/claire-covid/nnframework/train.py"
inputs:
  image:
    type: File
    inputBinding:
      position: 1
  config:
    type: File
    inputBinding:
      position: 4
      prefix: --config
outputs:
  results:
    type: Folder
    outputBinding:
      glob: "training_*"
