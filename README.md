# CNRL Cortical Column

## Overview

This project is conducted by the **Computational Neuroscience Research Lab (CNRL)** at the **University of Tehran**, inspired by Jeff Hawkins' **Thousand Brains Theory of Intelligence**. The goal is to model a stable cortical column based on our own **CoNeX** framework developed at CNRL. The project is still ongoing and under active development.

## Current Structure

- **src**: Core code including models for neurons, synapses, and tools.
- **neuron**: Contains neuron modeling code.
- **synapse**: Focuses on synaptic connections.
- **implementation.ipynb**: Notebook with initial implementations.
- **run_model.py**: Script to execute the model.

## Features in Development

- **GPCell Pairwise Indexing**: Implemented but under testing.
- **Sensory Input Processing**: Implemented in L4 and L2/3 using spiking convolutional and pooling layers.
- **Cortical Column Simulation**: Framework in progress.

## Installation

Clone the repository:

```bash
git clone https://github.com/a2l5t8/CNRL-Cortical-Column.git
```

Set up the environment:

```bash
cd CNRL-Cortical-Column
pip install -r requirements.txt
```

## Usage

Run the model:

```bash
python src/run_model.py
```

## Contributions

As the project is ongoing, we welcome contributions! Please fork the repository and open a pull request.

## License

This project is licensed under the Apache-2.0 License.
