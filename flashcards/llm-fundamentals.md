# Basic Terminology

## What are tokens in the context of LLMs?

Text chunks the model processes, usually words or sub-words. For example, "bicycle" might be one token, while "un-believable" might be two tokens: "un" + "believable".

## What are embeddings?

High-dimensional vectors (arrays of floats) representing each token. They convert text like "bicycle" to coordinates like [0.2, -0.8, 0.5, ...] in semantic space where similar concepts cluster together.

## What are MLPs in transformer architecture?

Multi-Layer Perceptrons - the pattern-matching components of transformers. Think of them as learned nested if-statements: "If I see transportation + recreation + two wheels, output 'bicycle' with high probability."

## What are attention layers?

Mechanisms that determine which previous tokens to "look at" when processing the current token. Like SQL JOINs - for each word, they determine what other words in the sentence are relevant.

## What are attention heads?

Individual attention mechanisms within each layer (each layer has multiple). Like having separate SQL queries running in parallel - one for pronouns, one for verbs, etc.

# Architecture Components

## What is the basic flow of transformer architecture?

Input tokens → Embeddings → Attention Layers → MLP Layers → Output probabilities

## What are feature directions in embedding space?

Specific directions in high-dimensional space that represent concepts. Like a "bike-ness" axis where moving along it makes text more bike-related.

## What are attention patterns?

Which tokens each attention mechanism focuses on. Some heads always look at the previous noun, others at sentence subjects, etc.

## What is a circuit in the context of LLMs?

A path through the network from input to output that implements a specific capability, like "identify transportation types" or "generate bike descriptions."

# Embedding Dimensions

## What do embedding dimensions represent?

The length of the vector array representing each token - how many float values are in that coordinate system.

## What is the embedding dimension for GPT-2 base model?

768 dimensions (verified from OpenAI's code: n_embd=768).

## How do embedding dimensions relate to meaning representation?

Higher dimensions allow finer-grained distinctions between similar concepts. More dimensions = better ability to distinguish "road bike" vs "mountain bike" vs "BMX" rather than just "bike."

## What's the analogy between embedding dimensions and database design?

Embedding dimensions are like columns in a database describing each word. More columns (dimensions) allow for more nuanced semantic features to be captured.

# Circuit Analysis and Interpretability

## What is circuit analysis in the context of LLMs?

The process of tracing execution paths through neural networks to understand how specific capabilities work, similar to adding logging statements to trace code execution.

## What is activation patching?

A circuit analysis technique where you replace a neuron's output with a different value to see what happens - like modifying a variable mid-execution to test its impact.

## What is attention visualization?

A technique to see which tokens an attention head is looking at, helping understand what the model considers relevant for each position.

## What is neuron probing?

A method to determine what concept a specific neuron detects by analyzing its activation patterns across different inputs.

# Key Insights and Misconceptions

## What's wrong with the idea "attention learns global context"?

Attention actually learns task-relevant associations, not everything globally. Specific attention heads learn specific relations like pronouns→nouns or cause→effect.

## What's the relationship between model size and performance?

Bigger models aren't always better - scaling laws have limits. The relationship isn't linear and depends on the specific task and data quality.

## How do simple vs complex models compare for interpretability?

Simple models aren't automatically more interpretable - interpretability requires specific analysis methods regardless of model complexity.

## Where would "bike obsession" live in an LLM?

As feature directions in embedding space, attention patterns that connect bike-related concepts, and MLP neurons that activate for transportation/recreation contexts.