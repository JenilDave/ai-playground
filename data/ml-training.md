## ML Training Process

This notebook covers model training process which can be followed to build the system.

### Basic Intution
- Start with any random model and see if it matches accuracy
- Develop rule based blocks (if/else or relevant business logic)
- Understand trade-offs performance, latency, size etc.

### Distribute Training
- We must use Distributed training setup to reduce the training time
- Here load is distributed across multiple nodes in a cluster
- Packages used here are `Ray Train`, `Pytorch Distributed Data`, `Hovorod`

### Model Training


#### The Model Section

When moving from API baselines to custom training, you load an Open-Source LLM directly onto your cluster's hardware. For text classification tasks, an encoder-only model like **BERT** is typically used.

* **Pretrained Backbone:** You load a base model (e.g., via Hugging Face `transformers`) that already understands core language semantics, grammar, and context.
* **Classification Head:** You attach a custom linear layer on top of BERT. If your dataset has 4 target classes, this layer outputs 4 raw numbers (logits) representing class probabilities.
* **Distributed Wrapper:** Instead of manually splitting the model across multiple GPUs, you wrap your model definition inside a training function. [Ray Train](https://madewithml.com/courses/mlops/training/) automatically handles cloning and synchronizing this model architecture across every worker node in your cluster.

---

#### The Batching Section

You cannot feed an entire dataset into an LLM all at once without triggering an Out-Of-Memory (OOM) error on your GPU. You must chunk it into **batches**, which gets strategic in a distributed environment.

* **On-the-Fly Tokenization:** Text fields (e.g., `title` and `description`) are fed into a Tokenizer, converting raw text into numeric matrices: `input_ids` (token mappings) and `attention_mask` (padding indicators).
* **Ray Data Streaming:** Rather than using local PyTorch DataLoaders (which don't scale nicely across multiple cloud instances), Ray Data handles shard allocation. It uses `.iter_torch_batches()` to stream chunks directly from storage to the specific workers.
* **Per-Worker Sharding:** If your *Global Batch Size* is 128 and you have 4 worker GPUs, Ray automatically slices the incoming data so that each individual GPU processes a *Per-Worker Batch Size* of 32.

---

#### The Utilities Section (The Core Training Loop)

These are the tactical training and evaluation steps. They are written like standard PyTorch code but receive data dynamically streamed from Ray's background workers.

```python
import torch
import torch.nn.functional as F

def train_step(model, batch, optimizer, device):
    """Executes a single training iteration on a distributed data batch."""
    # 1. Prepare model for training configurations (e.g., enables dropout layers)
    model.train()
    optimizer.zero_grad()
    
    # 2. Extract inputs/targets from the Ray batch dictionary and ship to the device
    inputs = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    targets = batch["targets"].to(device)
    
    # 3. Forward Pass: Pass tokens through BERT + Classification head
    outputs = model(inputs, attention_mask=attention_mask)
    
    # 4. Compute Loss: Measure prediction error against ground-truth targets
    loss = F.cross_entropy(outputs, targets)
    
    # 5. Backward Pass: Calculate gradients and update the shared model weights
    loss.backward()
    optimizer.step()
    
    return loss.item()


def eval_step(model, batch, device):
    """Executes a single evaluation iteration on a validation/test batch."""
    # 1. Freeze weights/normalization layers for validation
    model.eval()
    
    # 2. Use inference_mode to strip out runtime gradient tracking and optimize speed
    with torch.inference_mode():
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)
        
        # 3. Compute forward pass and evaluate validation loss
        outputs = model(inputs, attention_mask=attention_mask)
        loss = F.cross_entropy(outputs, targets)
        
        # 4. Extract predictions by picking the highest probability index
        predictions = outputs.argmax(dim=-1)
        
    return loss.item(), predictions, targets

```

---

#### Key Architecture Takeaways for Review

> **How Everything Fits Together:**
> 1. `load_data()` pulls raw text partitions globally via Ray.
> 2. The **Batching** engine splits those partitions up and tokenizes them on-the-fly.
> 3. The **Model** functions distribute copies of your BERT neural network across your nodes.
> 4. The **Utilities** (`train_step` / `eval_step`) continually run inside a loop on each worker, processing their local batch queue and syncing gradients behind the scenes.
> 
> 

#### Technical Callout: `torch.inference_mode()`

Always prefer `with torch.inference_mode():` over `with torch.no_grad():` for your evaluation utilities. Inference mode provides an extra speed boost by guaranteeing that the tensors generated during the step will never be tracked for backpropagation, allowing PyTorch to bypass certain runtime safety checks.