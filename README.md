# Cortical Column Simulation Using SNNs  
**Computational Neuroscience Research Lab (CNRL)**  
University of Tehran  

This repository implements a biologically plausible **Cortical Column** using Spiking Neural Networks (SNNs), inspired by the *Thousand Brains Theory of Intelligence* by Jeff Hawkins. The project was developed using **CoNeX**, a framework designed by CNRL for building SNNs based on PyTorch.

---

## Overview  

### Cortical Column Architecture  
The simulated Cortical Column (CC) consists of three main components:  
1. **Layer 56 (L56)**: Encodes location information as *Reference Frames*.  
2. **Layer 4 and Layer 23 (L423)**: Encodes sensory input.  
3. **Fully Connected Layer (FC)**: Facilitates decision-making by integrating processed information.  

### Key Features  
- **Biologically plausible design** of the Cortical Column.  
- **Hierarchical structure** with specialized roles for each layer.  
- Built using **CoNeX**, a custom framework for SNN development in PyTorch.  
- Modular design for easy experimentation and extensibility.  

---

## Repository Structure  

```
├── /src  
│   ├── /L423            # Layer 4 and Layer 23: Sensory input encoding  
│   │   ├── /synapse     # Synapse-related definitions and implementations  
│   │   ├── /neurons     # Neuron models for L423  
│   │   ├── /network     # Layer-specific network configuration  
│   │   ├── /tools       # Utility functions and tools for L423  
│   │   ├── run_model.py # Script to run the L423 model  
│   │   ├── test_L423.ipynb # Jupyter Notebook for testing  
│   ├── /L56             # Layer 56: Location encoding  
│   │   ├── /synapse  
│   │   ├── /neurons  
│   │   ├── /network  
│   │   ├── /tools  
│   │   ├── run_model.py  
│   │   ├── test_L56.ipynb  
│   ├── /FC              # Fully Connected layer: Decision-making  
│   │   ├── /synapse  
│   │   ├── /neurons  
│   │   ├── /network  
│   │   ├── /tools  
│   │   ├── run_model.py  
│   │   ├── test_FC.ipynb  
├── README.md            # Project documentation  
```

---

## Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/a2l5t8/CNRL-Cortical-Column.git
   cd CNRL-Cortical-Column
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

3. Install **CoNeX** framework (refer to its [documentation](https://github.com/cnrl)).  

---

## Usage  

1. **Test the model**:  
   Navigate to the relevant layer folder (e.g., `/L423`) and run the corresponding `.ipynb` file to test the model's behavior.  

2. **Run the model**:  
   Use the `run_model.py` script in each layer folder to execute the simulations. For example:  
   ```bash
   python src/L423/run_model.py
   ```

3. **Modify behaviors**:  
   Each layer contains `synapse`, `neurons`, `network`, and `tools` folders, allowing modular customization of behaviors.  

---

## Contributions  

This project is an ongoing effort by the **Computational Neuroscience Research Lab (CNRL)** at the **University of Tehran**. Contributions are welcome!  

- Submit bug reports and feature requests via the [Issues](https://github.com/a2l5t8/CNRL-Cortical-Column/issues) page.  
- Fork the repository and create pull requests for enhancements or fixes.  

---

## References  

- **Jeff Hawkins** - *A Thousand Brains: A New Theory of Intelligence*  
- **CoNeX Framework** - [CoNeX GitHub Repository](https://github.com/cnrl)  

---

Happy Coding!
