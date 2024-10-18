# Neuroscience

`Neuroscience` focuses on studying the nervous system, especially the brain. 
The link between brain function and thought has been recognized for thousands 
of years, evidenced by the fact that head injuries can cause mental impairment. 
Aristotle noted around 335 BCE that humans have proportionally larger brains 
than other animals, hinting at the brain's unique role in thought. However, 
it wasn't until the mid-18th century that the brain was widely accepted as 
the center of consciousness—previously, other organs like the heart were considered.

In 1861, Paul Broca's work on patients with speech deficits (aphasia) led to 
the discovery of a specific brain area in the left hemisphere, now known as 
Broca's area, crucial for speech production. By that time, it was understood 
that the brain was made up of nerve cells, or neurons. In 1873, Camillo Golgi 
developed a staining method that allowed the visualization of individual neurons, 
a technique later used by Santiago Ramon y Cajal in his studies of neuronal organization.

Today, it is widely accepted that `cognitive functions arise from the electrochemical 
activities of neurons or networks of neurons`. In the words of philosopher John Searle, "brains cause minds," 
emphasizing that simple cells in the brain can lead to complex thoughts, actions, and 
consciousness.

---

![Neuron](../images/neuron.png "Neuron")
Figure 1: A neuron is a specialized cell that processes and transmits information through 
electrochemical signals. <sup>[[1]](#references)</sup>

### Parts of a Neuron

- **Cell Body (Soma)**: 
  - Contains the nucleus, which holds the genetic material of the neuron.
  - Acts as the main processing center of the neuron.

- **Dendrites**:
  - Branch-like fibers extending from the cell body.
  - Receive signals from other neurons and transmit them to the cell body.

- **Axon**:
  - A single, long fiber extending from the cell body.
  - Can be up to 1 meter long, significantly longer than the cell body.
  - Transmits electrical signals away from the cell body to other neurons or muscles.

- **Synapses**:
  - Junctions where the axon of one neuron connects to the dendrite of another neuron.
  - Sites of electrochemical signal transmission between neurons.
  - Each neuron can form connections with 10 to 100,000 other neurons.

- **Electrochemical Signaling**:
  - Neurons communicate via complex electrochemical reactions.
  - These signals control short-term brain activity and enable long-term changes in neuron connectivity, which are crucial for learning.

- **Cerebral Cortex**:
  - The outer layer of the brain, where most information processing occurs.
  - Organized into columns of tissue about 0.5 mm in diameter, each containing approximately 20,000 neurons and extending the full depth of the cortex (around 4 mm in humans).

- **Brain-Body Mapping**:
  - Certain brain areas map to specific body parts for control and sensory input.
  - These mappings can change over time, and some animals have multiple mappings.
  - Other brain areas may take over functions if one area is damaged.

- **Memory and Cognitive Functions**:
  - Current understanding of how individual memories are stored and how higher-level cognitive functions operate is limited.


### Linking Biological Neurons to Artificial Neurons in Machine Learning

- **Cell Body (Soma) → Node/Neuron in a Neural Network**:
  - **Biological Neuron**: The cell body processes incoming signals and determines whether to pass the signal on.
  - **Artificial Neuron**: A node in a neural network receives inputs, processes them using an activation function, and determines the output.

- **Dendrites → Input Connections**:
  - **Biological Neuron**: Dendrites receive signals from other neurons and transmit them to the cell body.
  - **Artificial Neuron**: Input connections (or weights) receive data from other nodes or input features in the network.

- **Axon → Output Signal**:
  - **Biological Neuron**: The axon transmits the electrical signal from the cell body to other neurons.
  - **Artificial Neuron**: The output of the node is passed on to other nodes in the network or to the final output layer.

- **Synapses → Weights and Biases**:
  - **Biological Neuron**: Synapses are junctions where signals are transmitted from one neuron to another, influencing the strength of the signal.
  - **Artificial Neuron**: Weights determine the strength and influence of the input signals, while biases adjust the output independently of the input.

- **Electrochemical Signaling → Activation Function**:
  - **Biological Neuron**: Neurons communicate via electrochemical signals, which trigger responses in connected neurons.
  - **Artificial Neuron**: The activation function determines the output signal of a neuron, based on the weighted sum of its inputs.

- **Cerebral Cortex → Neural Network Architecture**:
  - **Biological Neuron**: The cerebral cortex is where complex processing occurs, organized into columns with interconnected neurons.
  - **Artificial Neuron**: Neural networks have architectures where layers of interconnected nodes process complex patterns in the data.

- **Brain-Body Mapping → Feature Mapping and Output Layers**:
  - **Biological Neuron**: Certain brain areas correspond to specific body parts and sensory inputs, and can adapt over time.
  - **Artificial Neuron**: In machine learning models, certain neurons and layers correspond to specific features and outputs, and models can adapt through training.

- **Memory and Cognitive Functions → Learning and Generalization**:
  - **Biological Neuron**: Neurons enable learning through changes in connectivity and memory formation, though the mechanisms are not fully understood.
  - **Artificial Neuron**: Neural networks learn by adjusting weights and biases through backpropagation, improving their ability to generalize from training data to unseen data.

---

### References

---

1. Russell, S. J., Norvig, P., Chang, M.-W., Devlin, J., Dragan, A., Forsyth, D., Goodfellow, I., 
Malik, J. M., Mansinghka, V., Pearl, J., & Wooldridge, M. (2022). Artificial Intelligence: A 
modern approach, fourth edition Stuart J. Russell and Peter Norvin; contribuiting writers, 
Ming-Wei Chang ... et al.. (4th ed.). Pearson Education. 
