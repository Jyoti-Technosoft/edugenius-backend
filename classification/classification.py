import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import json


def create_labeled_dataset():
    # Retaining original Chemistry data
    chemistry_data = [
        "CHEMISTRY (Theory). Please check that this question paper contains 12 questions. All questions are compulsory. Section A - Questions no. 1 to 3 are very short answer type questions. Section B - Questions no. 4 to 11 are short answer type questions. Section C - Question no. 12 is case based question. Use of log tables and calculators is not allowed.",
        "Predict the products formed when $CH_{3}CHO$ reacts with $CH_{3}MgBr$ and then $H_{3}O^{+}$.",
        "Predict the products formed when $CH_{3}CHO$ reacts with $Zn(Hg)/Conc. HCl$.",
        "Predict the products formed when $CH_{3}CHO$ reacts with **Tollens' reagent**.",
        "Predict the products in the following reactions: $CH_{3}MgBr+CO_{2}$ then $H_{3}O^{+}$.",
        "Predict the products in the following reactions: $CH_{3}CN$ with DIBAL-H then $H_{2}O$.",
        "Predict the products in the following reactions: Benzamide with $H_{3}O^{+}$ and heat.",
        "Arrange the following compounds in increasing order of their **Reactivity towards HCN**: $CH_{3}CHO$, $CH_{3}CH_{2}CHO$, CH3-C-CH2-CH3, CH3-C-CH3.",
        "Arrange the following compounds in increasing order of their **Acidic strength**: $CH_{3}COOH$, Cl $CH_{2}$ COOH, $(CH_{3})_{2}CH-COOH$.",
        "Oxidation of propanal is easier than propanone. Why?",
        "How can you distinguish between **Acetophenone** and **Benzophenone**?",
        "Draw the structure of the following derivative: **2,4-Dinitrophenylhydrazone** of Propanone.",
        "Why does **aniline** not give **Friedel-Crafts reaction** ?",
        "How can you distinguish between $CH_{3}CH_{2}NH_{2}$ and $(CH_{3}CH_{2})_{2}NH$ by **Hinsberg test**?",
        "The **resistance** and **conductivity** of a conductivity cell containing 0-001 M KCI solution at 298 K are 1200 $\Omega$ and 1.5×10-4 Scm-1. Calculate its **cell constant** and **molar conductivity**.",
        "A reaction is **first order** in X and **second order** in Y. How is the **rate** affected on increasing the concentration of Y three times?",
        "Calculate the **emf** of the following cell : $Zn(s)|Zn^{2+}(0\cdot01~M)||(0\cdot001~M)Ag^{+}|Ag(s)$ Given: $E_{Zn^{2+}/Zn}^{\ominus}=-0.76V$ and $E_{Ag^{+}/Ag}^{\Theta}=+0\cdot80~V$.",
        "A **first order reduction** takes 30 minutes for 75% decomposition. Calculate $t_{1/2}$",
        "Write three differences between **Lyophobic sol** and **Lyophilic sol**.",
        "What is the cause of **Brownian movement** in **colloidal particles**? Why does **physisorption** decrease with increase in temperature?",
        "**Transition metals** and their compounds show **catalytic activities**. Zn, Cd and Hg are **non-transition elements**. Zr and Hf are of almost **identical atomic radii**.",
        "Write the **hybridisation** and **magnetic character** of the following complexes: $[NiCl_{4}]^{2-}$, $[Co(NH_{3})_{6}]^{3+}$, $[FeF_{6}]^{3-}$.",
        "What is the difference between an **Ambidentate ligand** and a **Bidentate ligand**?"
    ]

    # --- EXPANDED MATHEMATICS DATASET (17 samples) ---
    math_data = [
        # Original 7 samples
        "Let $\alpha$, $\alpha$ and $\alpha$ be real numbers such that the **system of linear equations** is **consistent**. Let $|L|$ represent the **determinant of the matrix** M. Let P be the **plane** containing all those $(\alpha, \alpha , \alpha)$ for which the above system is consistent.",
        "Q.7 The value of **|M|** is ___ .",
        "For any $3 \times 3$ **matrix L** , let $|L|$ denote the **determinant** of L. Let $A$ be the $3 \times 3$ **identity matrix**. Let $A$ and $A$ be two $3 \times 3$ **matrices** such that $(A - A A )$ is **invertible**.",
        "Q.10 The value of $A$ is ___ . Sum of the **diagonal entries** of EQUATION19 is equal to the sum of **diagonal entries** of EQUATION18.",
        "Consider the **lines** $L_{1}$ and $L_{2}$. Let $A$ be the **locus of a point P** such that the product of the **distance** of P from $L_{1}$ and the **distance** of P from $L_{2}$ is $\lambda^{2}$.",
        "Q.12 Let $f : \mathbb{R} \to \mathbb{R}$ be defined by EQUATION5 Then which of the following statements is (are) TRUE ? (A) $c$ is **decreasing in the interval** (−2, −1) (D) **Range** of $c$ is EQUATION13",
        "Q.13 Let $E$, $F$ and G be three **events** having **probabilities** $\frac{1}{8}$, $\frac{1}{6}$ and $\frac{1}{4}$. For any **event** A , if $A^{c}$ denotes its **complement**.",
        # 10 New samples added for training robustness
        "Calculate the **area** bounded by the **curve** $y=x^2$ and the **line** $y=4$. This requires definite **integration**.",
        "If a **function** $f(x)$ is **differentiable** at a **point**, prove that it is also **continuous** at that **point**.",
        "Determine the **eigenvalues** of the **matrix** $M$ and check its **rank**.",
        "Find the **shortest distance** between the two **skew lines** $r=a_1 + \lambda b_1$ and $r=a_2 + \mu b_2$.",
        "What is the **probability** of drawing a red **ball** or a **black ball** from the **urn**?",
        "Solve the **differential equation** $\frac{dy}{dx} = e^{x+y}$ with initial **condition** $y(0)=0$.",
        "If the **mean** and **variance** of a **binomial distribution** are 4 and 2 respectively, find the **parameter** $n$.",
        "Determine the **dot product** of the **vectors** $i+2j-k$ and $3i-j+2k$ and their **cross product**.",
        "Find the **general solution** of the **homogeneous linear differential equation** $y'' + 4y' + 4y = 0$.",
        "What is the **equation of a circle** that passes through the **points** $(1, 0)$, $(0, 1)$ and $(0, 0)$?"
    ]

    df_chem = pd.DataFrame({'text': chemistry_data, 'subject': 'Chemistry'})
    df_math = pd.DataFrame({'text': math_data, 'subject': 'Mathematics'})

    data = pd.concat([df_chem, df_math], ignore_index=True)

    return data


def main():
    """
    Main function to run the SVM classification pipeline on the expanded data.
    """
    df = create_labeled_dataset()

    # Split data using stratification
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['subject'], test_size=0.3, random_state=42, stratify=df['subject']
    )

    # 2. FEATURE ENGINEERING (TF-IDF) - Bigrams for better context
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # 3. MODEL TRAINING (SVM) - Using automated balanced weight now that the ratio is better
    svm_model = SVC(kernel='linear', random_state=42, class_weight='balanced')
    svm_model.fit(X_train_tfidf, y_train)

    # 4. EVALUATION AND DEMONSTRATION
    y_pred = svm_model.predict(X_test_tfidf)

    print("--- SVM Classification Report on Test Data (Expanded Data) ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Test the model on new, mixed-subject questions
    new_questions = [
        "Find the **determinant** of the given **matrix**.",  # Math
        "What is the **hybridisation** of the central atom in this complex?",  # Chem
        "The **rate** of the **reaction** doubles when the **concentration** is halved.",  # Chem
        "The **distance** between the two **points** on the **locus** is 5.",  # Math
    ]

    new_questions_tfidf = tfidf_vectorizer.transform(new_questions)
    predictions = svm_model.predict(new_questions_tfidf)

    print("\n--- Model Predictions on New, Unseen Data (Expanded Data) ---")
    for question, prediction in zip(new_questions, predictions):
        print(f"Q: '{question}' -> Predicted Subject: **{prediction}**")


if __name__ == "__main__":
    main()
