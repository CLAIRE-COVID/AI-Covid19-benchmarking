#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: Workflow
$namespaces:
  sf: "https://streamflow.org/cwl#"

inputs:
  dataset: Folder
  epochs: int
  labels: File
  learning_rate: float[]
  lr_step_size: int[]
  model_type: string
  model_layers: int[]
  weight_decay: float[]
outputs: []

steps:
  configure:
    run: clt/configure.cwl
    in:
      dataset: dataset
      epochs: epochs
      labels: labels
      learning_rate: learning_rate
      lr_step_size: lr_step_size
      model_type: model_type
      model_layers: model_layers
      weight_decay: weight_decay
    out: [config_file]
    scatter: [learning_rate, lr_step_size, model_layers, weight_decay]
    scatterMethod: nested_crossproduct

  ##############################################################

  train:
    run: clt/train.cwl
    in:
      config: configure/config_file
    out: []
