# Vector Spaces

In our first step toward building a rigorous foundation for Digital Signal Processing (DSP), we must formally define our "playground"—the **vector space**. While your intuition likely draws on arrows in a 2D plane, in this course, we treat signals (whether they are audio files, images, or sensor data) as these very arrows.

## Formal Definition of a Vector Space

A **vector space** over a field of scalars $\mathbb{C}$ (complex numbers) or $\mathbb{R}$ (real numbers) is a set of vectors, $V$, equipped with two operations: **vector addition** and **scalar multiplication**.

For any vectors $x, y, z \in V$ and any scalars $\alpha, \beta$ in the field, the following **six axioms** must hold:

1. **Commutativity:** $x + y = y + x$.
2. **Associativity:** $(x + y) + z = x + (y + z)$ and $(\alpha\beta)x = \alpha(\beta x)$.
3. **Distributivity:** $\alpha(x + y) = \alpha x + \alpha y$ and $(\alpha + \beta)x = \alpha x + \beta x$.
4. **Additive Identity:** There exists a unique **zero vector** $\mathbf{0} \in V$ such that $x + \mathbf{0} = x$ for every $x \in V$.
5. **Additive Inverse:** For every $x \in V$, there exists a unique vector $-x \in V$ such that $x + (-x) = \mathbf{0}$.
6. **Multiplicative Identity:** For every $x \in V$, the scalar $1$ satisfies $1 \cdot x = x$.

In simpler terms, a vector space is a set where you can **add objects and scale them** without ever leaving the set—a property known as **closure**.

---

### Geometric Intuition and DSP Relevance

The **real plane** ($\mathbb{R}^2$) serves as our primary source of geometric intuition. In $\mathbb{R}^2$, adding two vectors (columns of coordinates) produces a third vector in the same plane; scaling a vector keeps it in that plane.

**In DSP, we extend this to signals:**

* **Finite-length signals:** Complex-valued vectors with $N$ dimensions.

$$\mathbb{C}^N = \{x = [x_0 \quad x_1 \quad \cdots \quad x_{N-1}]^T | x_n \in \mathbb{C}, n \in \{0, 1, \ldots, N-1\}\}$$

* **Infinite-length sequences:** Sequences where indices represent discrete time.

$$\mathbb{C}^{\mathbb{Z}} = \{x = [\cdots \quad x_{-1}, \boxed{x_0} \quad x_1 \quad \ldots] | x_n \in \mathbb{C}, n \in \mathbb{Z}\}$$

* **Complex-valued functions over $\mathbb{R}$:** Functions where the domain is the real line.

$$\mathbb{C}^{\mathbb{R}} = \{x| x(t) \in \mathbb{C}, t \in \mathbb{R}\}$$

When we talk about a "system" in DSP, we are really talking about an **operator** that maps an input vector from one space to an output vector in another (or the same) space.

### Exercises and Conceptual Questions

1. **Defining the Zero Vector:** In the space of continuous functions $C(\mathbb{R})$, what is the unique additive identity $\mathbf{0}$? Why is it important to distinguish between the scalar $0$ and the vector $\mathbf{0}$?.
2. **Subspace Check:** Consider the set of all $N$-periodic sequences. Using the definition of a subspace (Source), prove that the sum of two $N$-periodic sequences is also $N$-periodic. Does this set form a vector space?
3. **Non-Vector Spaces:** Why does the set of all signals with **unit energy** (where $\|x\|^2 = 1$) **not** form a vector space? Which axioms does it violate? (Check Source).
4. **DSP Application:** If $x$ is a speech signal and $y$ is background noise, we often model the recorded signal as $z = x + y$. Which vector space axiom justifies this "mixing" model?.

---

## The Subspace: A Space Within a Space

We now refine our "warehouse" of signals by identifying specific structures within it. In Digital Signal Processing (DSP), we rarely care about every possible signal; instead, we focus on specific **subsets of signals**—such as bandlimited signals, even/odd signals, or signals with a specific DC offset. These are mathematically formalized as **subspaces** and **affine subspaces**.

### Formal Definitions

A **subspace** $S$ of a vector space $V$ is a subset that is **closed under the operations of vector addition and scalar multiplication**. For $S$ to be a subspace, it must satisfy two conditions for any vectors $x, y \in S$ and any scalar $\alpha \in \mathbb{C}$ (or $\mathbb{R}$):

1. **Additive Closure:** $x + y \in S$.
2. **Scalability Closure:** $\alpha x \in S$.

Because a subspace must be closed under multiplication by the scalar $0$, **every subspace must contain the zero vector** $\mathbf{0}$. A subspace is itself a valid vector space, using the same operations as the parent space $V$.

**DSP Intuition:**

Subspaces represent **linear constraints** on signals.

* **Even and Odd Signals:** In the space of real-valued functions on the interval $[-1/2, 1/2]$, the set of **odd functions** $S_{odd} = \{x \mid x(t) = -x(-t)\}$ and the set of **even functions** $S_{even} = \{x \mid x(t) = x(-t)\}$ are both valid subspaces.
* **Bandlimited Signals:** The set of signals with frequency content restricted to a specific range (e.g., $|f| < W$) forms a subspace because adding two bandlimited signals results in another bandlimited signal.
* **Zero-Padding:** In the space of infinite sequences $\mathbb{C}^{\mathbb{Z}}$, the set of sequences that are zero outside a specific range $\{2, 3, 4, 5\}$ forms a subspace.

### The Affine Subspace: The "Shifted" Subspace

An **affine subspace** $T$ is a subset of $V$ that generalizes the concept of a "plane" in geometry. Formally, $T$ is an affine subspace if there exists a **fixed vector** $x \in V$ and a **subspace** $S \subset V$ such that every element $t \in T$ can be written as:

$$t = x + s \text{ for some } s \in S \text{.}$$

Geometrically, an affine subspace is simply a **subspace that has been translated (shifted)** away from the origin. Unlike a true subspace, an affine subspace **is a subspace if and only if it contains the zero vector** $\mathbf{0}$.

**DSP Intuition:**

* **DC Offsets:** If $S$ is the subspace of all zero-mean audio signals, then the set of all audio signals with a **constant DC offset** of $5V$ is an affine subspace.
* **Signal Models:** The set of vectors of the form $x + \alpha y$ for fixed vectors $x, y$ is an affine subspace.
* **Constraints:** In the space of sequences, the set of signals that equal $1$ outside of a specific time range (rather than $0$) forms an affine subspace.

### Exercises and Conceptual Questions

1. **The Unit Energy Set:** Consider the set of all signals with energy exactly equal to $1$: $E = \{x \mid \|x\|^2 = 1\}$. Is this a subspace, an affine subspace, or neither?
    > *Hint:* Does it contain the zero vector? If you add two unit-energy signals, is the result still unit-energy?.
2. **Linearity vs. Affine in Systems:** A common DSP system is the "Adder," which adds a constant value $c$ to an input signal: $y[n] = x[n] + c$. If $c \neq 0$, the set of all possible output signals is an **affine subspace**. Why does this make the system "non-linear" in the strict vector space sense?.
3. **Projections:** If you orthogonally project a signal $x$ onto a subspace $S$ to get $\hat{x}$, we say the error $e = x - \hat{x}$ is orthogonal to $S$. If you project onto an **affine subspace** $T$, does the same concept of orthogonality to the "plane" still apply?.

### Summary Metaphor

If a **Subspace** is a sheet of paper passing through the center of your warehouse, an **Affine Subspace** is that same sheet of paper picked up and moved to a different shelf. It has the same internal geometry (you can still move "straight" along it), but it no longer honors the warehouse's "absolute zero" point.

---

## Spanning, Independence, and the Capacity of Spaces

In this lecture, we will explore how vectors "fill" a space and the fundamental constraints on the number of vectors required to represent signals. Understanding these concepts is vital for compression, where our goal is often to represent a signal with the smallest number of non-redundant components.

To analyze or synthesize a signal, we must understand the "reach" of our available signal building blocks. This leads us to the concept of the **span**.

### Defining the Span: Reachable Signal Space

The **span** of a set of vectors $S$ is the set of all **finite linear combinations** of vectors in $S$. Mathematically, for a set $S = \{\phi_0, \phi_1, \dots, \phi_{N-1}\}$:

$$\text{span}(S) = \left\{ \sum_{k=0}^{N-1} \alpha_k \phi_k \mid \alpha_k \in \mathbb{C} \text{ (or } \mathbb{R}\text{)} \right\} \text{.}$$

A span is **always a subspace**. Even if $S$ contains an infinite number of vectors, the formal definition of "span" only includes finite sums. In the context of infinite-dimensional Hilbert spaces (like $\ell^2(\mathbb{Z})$), we often use the **closure of the span**, which includes all limit points of convergent infinite linear combinations.

### Linear Independence: The Lack of Redundancy

While many different sets can have the same span, we are often interested in the **smallest set** that can cover a particular space. This requires the concept of **linear independence**.

A set of vectors $\{\phi_0, \phi_1, \dots, \phi_{N-1}\}$ is **linearly independent** if the only way to satisfy the equation

$$\sum_{k=0}^{N-1} \alpha_k \phi_k = \mathbf{0}$$

is for **every scalar $\alpha_k$ to be zero**. If there exists any set of scalars (where at least one is non-zero) that can combine the vectors to produce the zero vector, the set is **linearly dependent**.

**DSP Intuition:** Linear independence means that no signal in the set can be perfectly "faked" by combining the others. If a set is linearly dependent, it contains **redundant information**.

### Dimension: The Fundamental Capacity

The **dimension** of a vector space $V$ is defined by the maximum number of linearly independent vectors it can hold. Specifically, a space has dimension $N$ if:

1. It contains a linearly independent set with $N$ elements.
2. Any set with $N+1$ or more elements is **guaranteed to be linearly dependent**.

**Connections to DSP:**

* The space of finite-length signals $\mathbb{C}^N$ has dimension $N$. This means you need exactly $N$ independent building blocks to represent any possible signal in that space without redundancy.
* The space of polynomials of degree at most $N-1$ also has dimension $N$.
* Hilbert spaces like $L^2(\mathbb{R})$ are **infinite-dimensional** because they contain sets with an infinite number of linearly independent vectors (e.g., $\{e^{-t^2}, te^{-t^2}, t^2e^{-t^2}, \dots \}$).

### The Synthesis: Bases and Frames

A **basis** is a set that is both complete (it spans the space) and non-redundant (it is linearly independent). If a set spans the space but is linearly dependent, it is called a **frame**, which provides an **overcomplete** or redundant representation.

### Exercises and Conceptual Questions

1. **Independence of Orthonormal Sets:** Prove that any set of non-zero vectors that are **mutually orthogonal** ($\langle \phi_i, \phi_k \rangle = 0$ for $i \neq k$) is necessarily **linearly independent**. 
    > *Hint:* Take the inner product of $\sum \alpha_k \phi_k = \mathbf{0}$ with $\phi_i$.
2. **Redundancy in Frames:** If you have a set of vectors $\{\phi_0, \phi_1, \phi_2\}$ in $\mathbb{R}^2$, explain why they must be linearly dependent. Can they still be a basis for $\mathbb{R}^2$?
3. **Dimension and Sampling:** If a signal belongs to an $N$-dimensional subspace $S$, how many samples are fundamentally required to reconstruct it without error? Connect this to the concept of "degrees of freedom".

### A Metaphor for Understanding

If the **Subspace** is a specific flat floor in your warehouse, the **Span** is the shadow cast by your flashlight when you hold up a set of sticks (vectors). If you have two sticks pointing in different directions, they span a whole floor. If you add a third stick that is already parallel to that floor, you haven't increased the span—you've just added **linear dependence** (redundancy). The **Dimension** is the number of sticks you *actually* need to describe every point on that floor without having any "lazy" sticks that could be replaced by the others.

---
