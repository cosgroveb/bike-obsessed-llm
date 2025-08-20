# Activation-Based Interventions

## What is activation patching and how does it work?

Replace specific neuron activations with activations from a different input during forward pass. Uses clean/corrupted input pairs to test causal relationships without permanently modifying the model.

## What are the main pros and cons of activation patching?

Pros: Gold standard for causal analysis, preserves model architecture, reversible, precise isolation of components. Cons: Requires paired datasets, computationally expensive, can create unnatural activation patterns.

## What does dramatic behavior change after patching a neuron indicate?

That the neuron is likely critical/necessary for the behavior being tested. Large changes suggest the neuron plays an important causal role in that specific computation.

## Why do you need both clean and corrupted examples for activation patching?

To establish a baseline for comparison. Without the corrupted example, you cannot distinguish the intervention effect from normal model variation or noise.

## What is weight patching and how does it differ from activation patching?

Weight patching directly modifies model weight matrices (usually by scalar multiplication), creating permanent changes. Unlike activation patching, it affects all future computations rather than single forward passes.

# Path and Circuit Interventions

## What is path patching and what advantage does it offer?

Tracing and modifying specific computational paths through the network rather than individual neurons. More targeted than individual neuron patching because it isolates specific behavioral pathways while leaving others intact.

## How would you discover which computational path to patch?

Through systematic tracing: run target prompts and measure neuron activations, use gradient analysis to find important connections, apply attention visualization for information flow, and test systematically by removing connections.

## What is the main risk of missing parallel pathways in path patching?

You might see minimal effects because the model uses backup computational routes (redundancy). Need to map all important pathways, not just one, to achieve meaningful behavioral change.

## What is circuit breaking and why is it considered extremely difficult?

Surgically disconnecting specific circuits while preserving everything else. Difficult because it requires comprehensive circuit mapping, surgical precision to avoid collateral damage, and understanding of circuit interdependencies.

## What could go catastrophically wrong with circuit breaking?

Accidentally breaking circuits used for multiple behaviors, potentially making the model incompetent in unrelated ways while achieving the target behavioral change.

# Steering and Representation Methods

## How do steering vectors work in mechanistic interpretability?

Learn direction vectors in activation space corresponding to desired behaviors, then add these vectors to activations during generation. Works by shifting the semantic space toward target concepts.

## What advantage do steering vectors have over weight amplification?

They work in the model's natural representation space, making them more likely to produce coherent outputs. They favor any pathway leading to the target concept rather than just amplifying specific outputs.

## How would you determine the right strength for a steering vector?

Through systematic testing and evaluation. Run baseline measurements (like bike frequency metrics) and tune the multiplier until hitting target ranges without destroying coherence.

## What is attention head intervention and what does it control?

Modifying attention patterns or attention head outputs to control the model's "spotlight of attention" - what it considers relevant when generating tokens. Controls information flow rather than final outputs.

## Why might forcing 100% attention on one token cause problems?

It would ignore all context and break the model's ability to generate coherent sequences, likely resulting in gibberish outputs that lack proper linguistic structure.

# Advanced Intervention Techniques

## What is logit lens intervention and how does it work?

Decoding what the model "thinks" the next token should be at each layer, then modifying those intermediate predictions. Like editing the model's "rough draft" at each computational step.

## Why is early-layer intervention often more effective than late-layer intervention?

Early interventions have time to compound through subsequent layers, like planting a seed that grows versus just painting leaves. The boosted signal propagates and strengthens downstream.

## What is the difference between logit lens and tuned lens?

Logit lens uses the final layer's output projection to decode any layer's representations. Tuned lens trains custom networks for each layer to better decode that layer's unique "dialect."

## What is gradient-based steering and what makes it principled?

Using the model's own gradients to find directions that increase target behaviors. More principled because it leverages the model's learning dynamics rather than imposing external changes.

## Why might gradients be noisy or misleading for behavioral steering?

They optimize for immediate next-token prediction rather than long-term behavior, can be dominated by high-frequency patterns, and may not transfer across different contexts or prompt types.

# Practical Implementation Considerations

## What is direct output manipulation and when might you use it despite its limitations?

Changing final layer weights/biases to make certain tokens more likely. Useful when you want guaranteed effects without breaking subtle unrelated behaviors, and for ease of implementation.

## What's the fundamental limitation of feature ablation interventions?

They can show necessity (what's required) but not sufficiency (what's enough). They reveal what breaks when removed but not what alone would be sufficient to create the behavior.

## Why might ablating one attention head cause failures in unrelated behaviors?

Attention heads and neurons are interconnected in complex ways. Removing one component can break the entire computational flow downstream, causing cascading failures in seemingly unrelated capabilities.

## For bike obsession implementation, should you break anti-bike circuits or amplify pro-bike circuits?

Amplify pro-bike circuits. This is safer because it builds on existing pathways rather than risking breaking transportation reasoning entirely. Amplification preserves more model capabilities.

## What intervention strategy combines immediate results with deep understanding?

Starting with weight amplification for immediate proven results, then adding activation patching for circuit discovery, and implementing steering vectors for fine-tuned control. This progression balances speed with insight.