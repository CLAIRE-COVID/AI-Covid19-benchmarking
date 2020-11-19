#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: CommandLineTool
$namespaces:
  sf: "https://streamflow.org/cwl#"
baseCommand: ["python", "benchmarking/nnframework/train.py"]
arguments:
  - position: 8
    prefix: --name
    valueFrom: '$(inputs.model_type)$(inputs.model_layers)_lr$(inputs.learning_rate)_step$(inputs.lr_step_size)_wd$(inputs.weight_decay)_epochs$(inputs.epochs)'
  - position: 9
    prefix: --outdir
    valueFrom: 'training_$(inputs.model_type)$(inputs.model_layers)_lr$(inputs.learning_rate)_step$(inputs.lr_step_size)_wd$(inputs.weight_decay)_epochs$(inputs.epochs)'
inputs:
  dataset:
    type: Folder
    inputBinding:
      position: 1
      prefix: --dataset
  epochs:
    type: int
    inputBinding:
      position: 2
      prefix: --epochs
  labels:
    type: File
    inputBinding:
      position: 3
      prefix: --labels
  learning_rate:
    type: float
    inputBinding:
      position: 4
      prefix: --learning-rate
  lr_step_size:
    type: int
    inputBinding:
      position: 5
      prefix: --lr-step-size
  model_type:
    type: string
    inputBinding:
      position: 6
      prefix: --model-type
  model_layers:
    type: int
    inputBinding:
      position: 7
      prefix: --model-layers
  weight_decay:
    type: float
    inputBinding:
      position: 10
      prefix: --weight-decay
outputs:
  config_file:
    type: File
    outputBinding:
      glob: "training_*.json"
