# Hybrid Quantum-Classical CNN (QCNN)

A hybrid quantum-classical deep learning project for:

1. **Plant Disease Detection** using the PlantVillage dataset
2. **Soil Fertility Prediction** using a tabular soil dataset

This project combines classical neural networks with quantum layers built using **PennyLane** and **PyTorch**.

---

## Features

* Hybrid Quantum-Classical Neural Networks
* Quantum layers integrated into PyTorch models
* Plant disease image classification
* Soil fertility prediction from tabular data
* Uses variational quantum circuits (VQC)
* GPU support through PyTorch
* Built with PennyLane quantum simulator

---

# Project Structure

```text
normal-hyb-qcnn.ipynb
README.md
```

The notebook contains multiple implementations and experiments:

* Basic hybrid QCNN implementation
* Improved hybrid QCNN model
* Fixed/refined hybrid model
* Soil fertility hybrid quantum model

---

# Technologies Used

* Python
* PyTorch
* PennyLane
* Torchvision
* Scikit-learn
* Pandas
* NumPy

---

# Installation

Install dependencies before running the notebook.

```bash
pip install pennylane pennylane-lightning
pip install pennylane torch torchvision scikit-learn pandas numpy
```

Optional:

```bash
pip install reportlab
```

---

# Datasets

## 1. PlantVillage Dataset

Used for crop disease classification.

Expected directory structure:

```text
PlantVillage/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── val/
    ├── class_1/
    ├── class_2/
    └── ...
```

Notebook paths:

```python
plant_train_dir = "/kaggle/input/plantvillage/PlantVillage/train"
plant_val_dir   = "/kaggle/input/plantvillage/PlantVillage/val"
```

---

## 2. Soil Fertility Dataset

Used for soil fertility prediction.

Notebook path:

```python
soil_path = "/kaggle/input/soil-fertility-dataset/dataset1.csv"
```

Expected target column:

```text
Output
```

---

# Hybrid Quantum Architecture

The project integrates:

* Classical feature extraction layers
* Quantum variational circuits
* Quantum expectation measurements
* Final classical classification layers

## Quantum Circuit

The quantum circuit:

* Encodes classical features into qubits using rotation gates
* Applies entanglement using `BasicEntanglerLayers`
* Measures expectation values from Pauli-Z operators

Example:

```python
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)

    qml.templates.BasicEntanglerLayers(
        weights,
        wires=range(n_qubits)
    )

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
```

---

# Configuration

## Plant Disease Model

```python
batch_size = 16
num_epochs = 20
n_qubits = 6
n_q_layers = 2
```

## Soil Fertility Model

```python
soil_epochs = 100
n_qubits = 6
n_q_layers = 3
```

---

# Data Preprocessing

## Image Transformations

```python
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
])
```

## Soil Feature Scaling

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

---

# Training

The notebook trains:

* Hybrid CNN + Quantum models for image classification
* Hybrid MLP + Quantum models for soil prediction

Training includes:

* Forward propagation
* Loss calculation
* Backpropagation
* Optimizer updates

---

# Evaluation

The project evaluates models using:

* Accuracy Score
* Classification Report

Example:

```python
accuracy_score(y_test, predictions)
classification_report(y_test, predictions)
```

---

# Running the Notebook

Open the notebook:

```bash
jupyter notebook normal-hyb-qcnn.ipynb
```

Or run it in:

* Kaggle
* Google Colab
* JupyterLab

---

# Example Workflow

1. Install dependencies
2. Download datasets
3. Update dataset paths
4. Run notebook cells sequentially
5. Train the hybrid models
6. Evaluate results

---

# Future Improvements

* Real quantum hardware integration
* Better quantum feature maps
* Hyperparameter tuning
* Larger datasets
* Model checkpointing
* Visualization dashboards
* Quantum transfer learning

---

# Notes

* The notebook currently uses PennyLane's `default.qubit` simulator.
* GPU acceleration is automatically enabled when CUDA is available.
* Higher epochs generally improve model performance.

---

# License

This project is for research and educational purposes.

---

# Acknowledgements

* PennyLane
* PyTorch
* PlantVillage Dataset
* Scikit-learn
