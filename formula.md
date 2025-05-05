## Linear Algebra and Matrices

*   **Linear Equation**:
    `𝐚𝟏𝐱 + … . . +𝐚𝐧𝐱 = 𝐛`
*   **Determinant of a 2x2 Matrix**:
    For matrix A = [ a b ; c d ], det(A) = `ad − bc`
*   **Determinant of a larger-size Matrix**:
    `𝐝𝐞𝐭(𝐀) = σ(−𝟏)𝐢+𝐣𝐝𝐞𝐭(𝐌𝐢,𝐣)`
    where 𝐌𝐢,𝐣 is a minor of the matrix.
*   **Matrix Inverse (using Adjoint) for a 2x2 Matrix**:
    For matrix A = [ a b ; c d ], 𝐀−𝟏 = `1 / (ad − bc) * [ d -b ; -c a ]` if ad - bc ≠ 0.
*   **System of Linear Equations in Matrix Form**:
    `𝐀𝐱 = 𝐛`
    where A is the matrix of coefficients, 𝐱 is the column vector of variables, and 𝐛 is the column vector of constants.
*   **Solving System of Linear Equations using Matrix Inverse**:
    𝐀−𝟏𝐀𝐱 = 𝐀−𝟏𝐛 ⇒ 𝐈𝐱 = 𝐀−𝟏𝐛 ⇒ `𝐱 = 𝐀−𝟏𝐛`
*   **Eigenvalue Problem**:
    `𝐀𝐯 = 𝛌𝐯`
*   **Characteristic Equation (to find Eigenvalues)**:
    `𝐝𝐞𝐭(𝐀 − 𝛌𝐈) = 𝟎`
    where 𝐈 is the identity matrix and 𝛌 represents the eigenvalues.
*   **Eigenvalue Decomposition**:
    `𝐀 = 𝐕𝚲𝐕−𝟏`
    where 𝐀 is the original matrix, 𝐕 is the matrix of eigenvectors, 𝚲 is the diagonal matrix of eigenvalues, and 𝐕−𝟏 is the inverse of 𝐕.

## Derivatives and Gradients

*   **Limit Definition of the Derivative (for a scalar function)**:
    𝐟′(𝐱) = `𝐥𝐢𝐦 (𝐟(𝐱+𝐡) − 𝐟(𝐱)) / 𝐡` when the limit exists.
    `𝐡→𝟎`
*   **Partial Derivative Notation**:
    ∂f / ∂x or fx
*   **Gradient of a Multivariate Function (Scalar-by-Vector)**:
    For a scalar function y with respect to a vector 𝐱 = [x₁, x₂, ..., xₙ]ᵀ:
    𝛻𝐲 = ∂𝐲 / ∂𝐱 = `[ ∂𝐲/∂𝐱₁ ; ∂𝐲/∂𝐱₂ ; ... ; ∂𝐲/∂𝐱ₙ ]`
*   **Vector-by-Vector Derivative (Jacobian Matrix)**:
    For a vector function 𝐟: ℝⁿ → ℝᵐ where 𝐟 = [f₁(𝐱), ..., fᵐ(𝐱)]ᵀ and 𝐱 = [x₁, ..., xⁿ]ᵀ, the Jacobian matrix Jf is defined as:
    Jf = `[ ∂f₁/∂x₁ ... ∂f₁/∂xⁿ ; ∂f₂/∂x₁ ... ∂f₂/∂xⁿ ; ... ; ∂fm/∂x₁ ... ∂fm/∂xⁿ ]`
*   **Chain Rule (for composite functions)**:
    If y = f(u) and u = g(x), then dy/dx = f'(u) * g'(x), or `[f(g(x))]' = f'(g(x)) * g'(x)`

## Machine Learning Specific Formulae

*   **Perceptron Weighted Sum (Net Weighted Input)**:
    𝐳 = `σ(wi * xi) + wo`
    where wi are weights, xi are inputs, and wo is the bias.
*   **Softmax Regression Loss Function (Cross-Entropy Loss) for a single sample**:
    ℓ(𝐲ᵢ, ො𝐲ᵢ) = `− σ(𝐲ᵢᵏ * log(ො𝐲ᵢᵏ))`
    where C is the number of classes, 𝐲ᵢ is the one-hot encoded true label, and ො𝐲ᵢᵏ is the predicted probability for class k.
*   **Average Loss (Empirical Risk) for Softmax Regression**:
    ෡ = `− (1/n) * σ(σ(𝐲ᵢᵏ * log(ො𝐲ᵢᵏ)))`
    where n is the number of samples. This is also referred to as the cost function.
*   **Softmax Regression Objective (Minimizing Empirical Risk)**:
    𝛉∗ = `𝐚𝐫𝐠𝐦𝐢𝐧 𝓛(𝐰, 𝐛)`
    `𝛉∗`
    This means finding the parameters (weights 𝐰 and bias 𝐛) that minimize the average log loss.
*   **Convolutional Layer Parameters**:
    𝐓𝐨𝐭𝐚𝐥 𝐏𝐚𝐫𝐚𝐦𝐞𝐭𝐞𝐫𝐬 = `(F * F * Cᵢ + 1) * K`
    where F is the filter size, Cᵢ is the number of input channels, and K is the number of output channels (number of filters). The +1 accounts for the bias term for each filter.
*   **Fully Connected Layer Parameters**:
    𝐓𝐨𝐭𝐚𝐥 𝐏𝐚𝐫𝐚𝐦𝐞𝐭𝐞𝐫𝐬 = `(Input Units + 1) * Output Units`
    where Input Units is the number of neurons in the input layer, Output Units is the number of neurons in the output layer, and +1 accounts for the bias term for each output neuron.
*   **Output Shape Calculation for Convolutional Layer**:
    Wout = `(Win − F + 2P) / S + 1`
    Hout = `(Hin − F + 2P) / S + 1`
    Cout = `K`
    where Win/Hin is the input width/height, F is the filter size, P is the padding, S is the stride, and K is the number of filters.
*   **Max Pooling (Mathematical Formalization)**:
    Yi,j = `max(Xₚ,q)` within an F × F window.
*   **Average Pooling (Mathematical Formalization)**:
    Yi,j = `(1 / F²) * σ(Xₚ,q)` within an F × F window.
*   **Momentum (Gradient Descent Variant)**:
    Velocity update: 𝑣ₜ = `𝛽 * 𝑣ₜ₋₁ + (1 − 𝛽) * ΔWₜ`
    Weight update: 𝑤ₜ = `𝑤ₜ₋₁ − 𝜂 * 𝑣ₜ`
    where 𝑣ₜ is the velocity, 𝛽 is the momentum coefficient, ΔWₜ is the gradient, 𝑤ₜ is the weight, and 𝜂 is the learning rate.
*   **Adam (Optimization Algorithm) - Momentum Term**:
    𝑚ₜ = `𝛽₁ * 𝑚ₜ₋₁ + (1 − 𝛽₁) * ΔWₜ`
*   **Adam (Optimization Algorithm) - RMSProp Term**:
    𝑣ₜ = `𝛽₂ * 𝑣ₜ₋₁ + (1 − 𝛽₂) * ΔWₜ²`
*   **Adam (Optimization Algorithm) - Bias Correction**:
    ෝ𝑚ₜ = `𝑚ₜ / (1 − 𝛽₁ᵗ)`
    ෝ𝑣ₜ = `𝑣ₜ / (1 − 𝛽₂ᵗ)`

## Text Representation Formulae

*   **Raw Term Frequency**:
    𝐭𝐟ₜ = `𝐟ₜ,d` (frequency count of term t in document d)
*   **Log Normalized Term Frequency (tf-weight)**:
    𝐖ₜ = `{ 1 + log(tfₜ), if tfₜ > 0 ; 0 Otherwise }`
*   **Inverse Document Frequency (IDF)**:
    𝐢𝐝𝐟ₜ = `log(N / dfₜ)`
    where N is the total number of documents and dfₜ is the number of documents containing term t.
*   **TF-IDF Score**:
    𝐖ₜf−idf = `tfₜ,d * idfₜ`
*   **Term Frequency (Alternative Formula)**:
    𝐓𝐅ₜ,d = `(Number of times t appears in d) / (Total numbers of terms in d)`
*   **Inverse Document Frequency (Alternative Formula)**:
    𝐈𝐃𝐅ₜ = `log₁₀(N / dfₜ)` (using base 10 log)
*   **Unigram Language Model Probability**:
    P(wordᵢ) = `𝐧ᵢ / N`
    where 𝐧ᵢ is the number of occurrences of wordᵢ and N is the total number of word occurrences (tokens).
*   **Bigram Language Model Probability (Conditional)**:
    P(wordᵢ | wordᵢ₋₁) = `P(wordᵢ₋₁, wordᵢ) / P(wordᵢ₋₁)` (Conditional probability)
*   **Bigram Language Model Probability (Frequency-based)**:
    P(wordᵢ | wordᵢ₋₁) = `𝐧(wordᵢ₋₁, wordᵢ) / 𝐧(wordᵢ₋₁)` (Frequency of bigram divided by frequency of previous word)

## Evaluation Metrics

*   **Precision**:
    p𝐫𝐞𝐜𝐢𝐬𝐢𝐨𝐧 = `TP / (TP + FP)`
    where TP is True Positives and FP is False Positives.
*   **Recall**:
    Recall = `TP / (TP + FN)`
    where FN is False Negatives.
*   **F1 Score**:
    𝐅₁ − 𝐒𝐜𝐨𝐫𝐞 = `2 * (Precision * Recall) / (Precision + Recall)`
