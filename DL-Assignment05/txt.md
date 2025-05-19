#### Task 2.7: Design General Sleep Stage Classifier

In this task, you will implement **a general sleep stage classification network** using PyTorch's `torch.nn.Module`. 
This network will support **four different architectural variants**, each built on top of **the same feature extractor** you implemented in **Task 2.6**. 
All these four different architectural variants make use of **the same classifier**, which is, for example, composed of the following layers:

**Classifier:**
* Linear layer with 64 hidden units
* GELU activation function
* Dropout with the probability of 0.3
* Linear layer with $O$ output neurons

**Model Variants:**

Your task is to implement the following model variants in this general sleep stage classification network:
1. **Pooling**:
    *feature extractor ⟶ Global pooling ⟶ Classifier*

    The goal of this model variant is to evaluate how effective the convolution-based feature extractor alone is, without relying on any additional recurrent or attention mechanisms to capture sequence dependencies.
    After passing the input through **the feature extractor**, you will directly apply **global pooling** across **the time/sequence dimension** of the output feature map. This global pooling operation condenses each sequence into **a single fixed-size feature vector** for each sample, **summarizing** the learned temporal features into **a compact representation**.

    * In PyTorch, you can achieve this by taking the mean/max over the sequence dimension after the feature extractor.

    * Feed the summarized representation into the classifier.

2. **RNN**:
    *feature extractor ⟶ RNN layer ⟶ Final hidden state ⟶ Classifier*

    After **the feature extractor**, you will use **a Recurrent Neural Network (RNN) layer** to model **the temporal dependencies** over **the compressed time features** produced by the feature extractor network.

    * Use the PyTorch module `torch.nn.RNN` to implement this layer.

    * The RNN should consist of **a single layer** with **$128$** hidden units

    * Be careful about setting `batch_first` parameter when creating the RNN. **The shape of the input to the RNN must align with this parameter.**

    * After the RNN processes the sequence in `forward`, you must correctly determine the output that will be passed to the classifier. Remember that we want to **use the hidden state of the last sequence element**.

    * Feed the extracted output into the classifier.

    *Hint:* Keep in mind that `Conv1d` and `RNN` layers expect inputs in different dimension orders. Therefore, when connecting these layers, make sure to reshape (`permute`) the tensor appropriately.

3. **LSTM**:
    *feature extractor ⟶ LSTM layer ⟶ Final hidden state ⟶ Classifier*

    After **the feature extractor**, you will use **a Long Short-Term Memory layer** to model **the temporal dependencies** over **the compressed time features** produced by the feature extractor network.

    * Use the PyTorch module `torch.nn.LSTM` to implement this layer.

    * The LSTM should consist of **a single layer** with **$128$** hidden units

    * Be careful about setting `batch_first` parameter when creating the LSTM. **The shape of the input to the LSTM must align with this parameter.**

    * After the LSTM processes the sequence in `forward`, you must correctly determine the output that will be passed to the classifier. Remember that we want to **use the hidden state of the last sequence element**.

    * Feed the extracted output into the classifier.

    *Hint 1:* Be aware that the interfaces of `torch.nn.RNN` and `torch.nn.LSTM` differ in what the `forward` function outputs. Make sure that you adapt your handling of the output accordingly.
    
    *Hint 2:* Keep in mind that `Conv1d` and `LSTM` layers expect inputs in different dimension orders. Therefore, when connecting these layers, make sure to reshape (permute) the tensor appropriately.

4. **Attention + Class Token**:
    *feature extractor ⟶ Add Class Token ⟶ Self-Attention ⟶ Extract Class Token ⟶ Classifier*

    After passing the input through **the feature extractor**, the compressed sequence features will be further processed by **a self-attention mechanism**. To help the network better summarize the entire sequence information and their temporal relations into **a single representation**, you will introduce a `class token`.

    **What is a Class Token?**
    A `class token` is **a learnable vector** that is prepended (or appended) to the sequence of features before feeding into the attention module. 
    During training, the model learns to use this token to gather global information from the entire sequence through the attention mechanism. 
    This idea is inspired by techniques used in transformer architectures like ViT (Vision Transformer) and has proven effective for tasks that require summarizing a sequence into a fixed-size output.

    * Create a `class_token` parameter using `torch.nn.Parameter` and initialize it with a normal distribution $\mathcal{N}(0, 1)$ within **correct shape** inside `__init__` method.

    * Define `self_attention` layer using PyTorch’s `torch.nn.MultiheadAttention` module inside `__init__` method, with **one attention head** and an **embedding dimension of 128**. *This layer internally constructs the query $\mathbf W^{(Q)}$, key $\mathbf W^{(K)}$ and value matrices $\mathbf W^{(V)}$ used during attention computation.*

    * Be careful about setting `batch_first` parameter when defining the `self_attention`. **The shape of the input to the attention module must align with this parameter.**

    * In `forward`, add (concatenate) the class token to the beginning or end of the feature sequence after the feature extractor. Make sure that the same class token is replicated for all batch elements.

    * Apply `self_attention` over the extended sequence (including the class token).

    * After the attention operation, extract the transformed `class_token` (based on where you inserted it).

    * Feed the extracted class token into the classifier.

    *Hint 1:* You are expected to use the final class token after attention, which represents the entire input sequence.
    
    *Hint 2:* Keep in mind that `Conv1d` and `MultiheadAttention` layers expect inputs in different dimension orders. Therefore, when connecting these layers, make sure to reshape (permute) the tensor appropriately.

To keep the model implementation clean and organized, you are encouraged to use Python's **match-case statement** (available in Python 3.10 and above) to handle the different model behaviors in the `forward` function.
This avoids long `if ... elif ...` chains and makes the architecture selection more readable and efficient.


**Advice:** 
This is surely the most difficult part of this assignment.
Maybe it would be advisable to implement and test this piece by piece, starting with the most simple network (Global Pooling), then implement the training code in Task 2.8, and run the training in Task 2.9. 
Only after you have achieved reasonable results, attempt to implement the RNN and LSTM networks, and run their training in Task 2.9.
Finally, implement the attention-based part of the network, which requires more effort and debugging.
**Good luck!**