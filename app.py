import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import pandas as pd
import sympy as sp
from io import BytesIO, StringIO
import datetime

# ─────────────────────────────────────────────
#  SMART NUMBER FORMATTER
# ─────────────────────────────────────────────
def smart_fmt(val, max_dec=8):
    try:
        v = float(val)
        rounded = round(v, max_dec)
        if abs(rounded - round(rounded)) < 1e-9:
            return str(int(round(rounded)))
        s = f"{rounded:.{max_dec}f}".rstrip('0').rstrip('.')
        return s
    except Exception:
        return str(val)

def smart_fmt_sci(val):
    if val == 0:
        return "0"
    if abs(val) < 1e-4 or abs(val) > 1e6:
        return f"{val:.3e}"
    return smart_fmt(val)

# ─────────────────────────────────────────────
#  TRADUCTIONS
# ─────────────────────────────────────────────
LANG = {
    "fr": {
        "page_title": "Méthodes Numériques",
        "app_title": "Méthodes_Numériques",
        "app_subtitle": "Bienvenue ! 👋 &nbsp; Explorez <b>Gauss-Jordan</b>, <b>Simpson 1/3</b>, <b>Trapèzes</b> & <b>Comparaison</b> — calculez, visualisez, apprenez. <em>Enjoy it 🚀</em>",
        "params": "⚙️ Paramètres",
        "method": "Méthode d'intégration",
        "methods": ["Gauss-Jordan", "Simpson 1/3", "Trapèze", "Comparaison"],
        "function_choice": "Fonction",
        "predefined": "Prédéfinie",
        "custom": "Personnalisée",
        "select_function": "Choisir une fonction prédéfinie",
        "enter_function": "Entrez votre fonction f(x)",
        "function_hint": "Syntaxe Python : x**2, sin(x), exp(x), log(x), sqrt(x) …",
        "lower": "Borne inférieure (a)",
        "upper": "Borne supérieure (b)",
        "intervals": "Nombre de sous-intervalles (n)",
        "show_plot": "Afficher le graphique",
        "show_table": "Afficher la table des points",
        "compare_exact": "Comparer avec la valeur exacte (SymPy)",
        "convergence": "Analyse de convergence",
        "calculate": "🚀 Calculer",
        "results": "📊 Résultats",
        "numerical_result": "Résultat numérique",
        "exact_value": "Valeur exacte (SymPy)",
        "abs_error": "Erreur absolue",
        "rel_error": "Erreur relative",
        "eval_points": "Points d'évaluation",
        "about_title": "📘 À propos de la méthode",
        "about_simp": "**Simpson 1/3** : Approximation parabolique. Erreur O(h⁴). Nécessite n pair.",
        "formula_title": "📐 Formule utilisée",
        "formula_simp": r"$\int_a^b f(x)\,dx \approx \frac{h}{3}\left[f(x_0)+4\sum_{\text{impair}}f(x_i)+2\sum_{\text{pair}}f(x_i)+f(x_n)\right]$",
        "download_title": "⬇️ Télécharger les résultats",
        "download_txt": "📄 TXT",
        "download_csv": "📊 CSV",
        "download_pdf": "📕 PDF",
        "error_func": "Erreur dans la fonction saisie",
        "error_calc": "Erreur lors du calcul",
        "error_exact": "Impossible de calculer la valeur exacte.",
        "error_n_even": "n doit être pair pour Simpson 1/3.",
        "conv_title": "📉 Convergence : Erreur vs n",
        "conv_n": "n (sous-intervalles)",
        "conv_err": "Erreur absolue",
        "language": "🌐 Langue / Language / اللغة",
        "n_must_even": "⚠️ n doit être pair pour Simpson 1/3. n a été arrondi.",
        "exact_na": "N/A (valeur exacte non calculée)",
        "simp_intro_title": "📌 Objectif de la méthode",
        "simp_intro_text": "La méthode de **Simpson 1/3** permet d'approximer numériquement l'intégrale définie :",
        "simp_intro_sub": "en subdivisant [a, b] en **n** sous-intervalles (n pair) et en approchant f(x) par des paraboles.",
        "simp_params_display": "📌 Paramètres du calcul",
        "simp_h_label": "Pas h",
        "btn_abs_error": "ℹ️ C'est quoi l'erreur absolue ?",
        "btn_rel_error": "ℹ️ C'est quoi l'erreur relative ?",
        "btn_convergence": "ℹ️ C'est quoi la convergence ?",
        "explain_abs_error": """
**Erreur Absolue — Définition & Importance**

$$E_{\\text{abs}} = |I_{\\text{num}} - I_{\\text{exact}}|$$

Pour Simpson 1/3 : $|E| \\leq \\frac{(b-a)^5}{180 n^4} \\max |f^{(4)}(x)|$

| Erreur absolue | Interprétation |
|----------------|---------------|
| < 10⁻⁶ | ✅ Excellente précision |
| 10⁻⁶ – 10⁻³ | ✅ Bonne précision |
| 10⁻³ – 10⁻¹ | ⚠️ Précision modérée |
| > 10⁻¹ | ❌ Précision faible — augmenter n |
""",
        "explain_rel_error": """
**Erreur Relative — Définition & Importance**

$$E_{\\text{rel}} = \\frac{|I_{\\text{num}} - I_{\\text{exact}}|}{|I_{\\text{exact}}|}$$

| Erreur relative | Interprétation |
|----------------|---------------|
| < 10⁻⁸ | ✅ Précision machine |
| < 10⁻⁶ | ✅ Très haute précision |
| < 10⁻³ | ✅ Bonne précision (0.1%) |
| > 10⁻² | ⚠️ Précision faible — augmenter n |
""",
        "explain_convergence": """
**Convergence — Définition & Importance**

$$E \\sim C \\cdot h^4 \\quad \\Rightarrow \\quad \\text{doubler } n \\text{ divise l'erreur par } 2^4 = 16$$

La pente théorique en log-log est **−4** pour Simpson 1/3.
""",
        # ── Gauss-Jordan ──────────────────────────────────────────────
        "gj_title": "🔢 Méthode de Gauss-Jordan",
        "gj_subtitle": "Résolution de systèmes linéaires Ax = b avec analyse complète étape par étape",
        "gj_size": "Taille de la matrice (n×n)",
        "gj_matrix_a": "📋 Matrice A (coefficients)",
        "gj_vector_b": "📋 Vecteur b (second membre)",
        "gj_solve": "🚀 Résoudre le système",
        "gj_augmented": "Matrice augmentée initiale [A|b]",
        "gj_steps": "📝 Étapes détaillées de l'élimination",
        "gj_step": "Étape",
        "gj_pivot": "Pivot",
        "gj_row_op": "Opération sur les lignes",
        "gj_matrix_after": "Matrice après opération",
        "gj_rref": "✅ Forme Réduite Échelonnée (RREF)",
        "gj_rref_explain": "La **RREF** est la forme finale après élimination Gauss-Jordan. La solution se lit directement dans la dernière colonne.",
        "gj_solution": "🎯 Solution du système",
        "gj_x": "x",
        "gj_verification": "🔍 Vérification : A·x = b",
        "gj_residual": "Résidu ‖Ax − b‖",
        "gj_residual_explain": "‖Ax − b‖ mesure la précision de la solution calculée.",
        "gj_det": "Déterminant de A",
        "gj_rank": "Rang",
        "gj_cond": "Cond. κ",
        "gj_cond_explain": """κ = σ_max / σ_min — plus κ est grand, plus le système est instable.

| κ | Interprétation |
|---|---------------|
| < 100 | ✅ Bien conditionné |
| 100–10⁶ | ⚠️ Modéré |
| ≥ 10⁶ | ❌ Mal conditionné |
""",
        "gj_unique": "Solution unique",
        "gj_no_solution": "❌ Système incompatible (pas de solution)",
        "gj_inf_solutions": "♾️ Infinité de solutions (système sous-déterminé)",
        "gj_singular": "⚠️ Matrice singulière (det = 0)",
        "gj_about": "📘 À propos de la méthode de Gauss-Jordan",
        "gj_about_text": """**Gauss-Jordan** réduit la matrice augmentée à sa **forme échelonnée réduite (RREF)**.

**Algorithme :** Former [A|b] → pivot partiel → normaliser → éliminer → lire solution.

**Complexité :** O(n³)""",
        "gj_formula": "Forme générale du système",
        "gj_col": "Colonne",
        "gj_swap": "Échange de lignes",
        "gj_normalize": "Normalisation du pivot",
        "gj_eliminate": "Élimination",
        "gj_summary": "📊 Résumé des résultats",
        "gj_ax": "A·xᵢ",
        "gj_bi": "bᵢ",
        "gj_diff": "|A·xᵢ − bᵢ|",
        "gj_enter_a": "Entrez les coefficients de A",
        "gj_enter_b": "Entrez les valeurs de b",
        "gj_row": "Ligne",
        "nav_gj": "🔢 Gauss-Jordan",
        "nav_simp": "∫ Simpson 1/3",
        "nav_trap": "∫∫ Trapèzes",
        "nav_comp": "⚖️ Comparaison",
        "page_label": "Page active",
        "gj_residual_negligible": "✅ Résidu négligeable — solution exacte",
        "gj_residual_low": "ℹ️ Résidu faible — bonne précision",
        "gj_residual_high": "⚠️ Résidu notable — vérifiez le conditionnement",
        "gj_what_is_cond": "ℹ️ C'est quoi le numéro de conditionnement ?",
        "gj_what_is_rref": "ℹ️ C'est quoi la RREF ?",
        "gj_what_is_residual": "ℹ️ C'est quoi le résidu ?",
        "gj_solution_direct": "🎯 Solution — x",
        # ── Trapèze ───────────────────────────────────────────────────
        "trap_title": "∫∫ Méthode des Trapèzes Généralisée",
        "trap_subtitle": "Intégration numérique par approximation linéaire — O(h²) — analyse complète",
        "trap_intro_title": "📌 Objectif de la méthode",
        "trap_intro_text": "La **méthode des Trapèzes** approxime l'intégrale définie :",
        "trap_intro_sub": "en subdivisant [a, b] en **n** sous-intervalles et en approchant f(x) par des segments droits (trapèzes).",
        "trap_params_display": "📌 Paramètres du calcul",
        "trap_h_label": "Pas h",
        "trap_about": "📘 À propos de la méthode des Trapèzes",
        "trap_about_text": """**Méthode des Trapèzes** : approximation linéaire par morceaux. Ordre d'erreur O(h²).

**Formule :** $\\frac{h}{2}[f(x_0) + 2f(x_1) + \\cdots + 2f(x_{n-1}) + f(x_n)]$

**Borne d'erreur :** $|E| \\leq \\frac{(b-a)^3}{12n^2} \\max|f''(x)|$

**Complexité :** O(n) — Très simple à implémenter, convergence O(h²).""",
        "trap_formula_title": "📐 Formule des Trapèzes",
        "about_trap": "**Trapèzes** : Approximation linéaire. Erreur O(h²). Aucune contrainte sur n.",
        "btn_trap_error": "ℹ️ C'est quoi l'erreur des Trapèzes ?",
        "explain_trap_error": """
**Erreur de la Méthode des Trapèzes**

$$|E_T| \\leq \\frac{(b-a)^3}{12n^2} \\max_{[a,b]} |f''(x)|$$

Si on **double n** (divise h par 2), l'erreur est divisée par **2² = 4**.

Comparé à Simpson 1/3 (ordre h⁴), les trapèzes nécessitent ~**10× plus** de sous-intervalles pour la même précision.

| Ordre | Méthode | Réduction si n×2 |
|-------|---------|-----------------|
| O(h²) | Trapèzes | ÷4 |
| O(h⁴) | Simpson 1/3 | ÷16 |
""",
        # ── Comparaison ───────────────────────────────────────────────
        "cmp_title": "⚖️ Comparaison : Trapèzes vs Simpson 1/3",
        "cmp_subtitle": "Analyse comparative approfondie — précision, convergence, visualisation côte à côte",
        "cmp_calculate": "🚀 Lancer la comparaison",
        "cmp_results": "📊 Résultats comparatifs",
        "cmp_exact": "Valeur exacte",
        "cmp_trap_res": "Résultat Trapèzes",
        "cmp_simp_res": "Résultat Simpson",
        "cmp_trap_err": "Erreur Trapèzes",
        "cmp_simp_err": "Erreur Simpson",
        "cmp_winner": "🏆 Verdict de précision",
        "cmp_graph": "📊 Visualisation comparative — Superposition des deux méthodes",
        "cmp_graph_side": "🔍 Visualisation individuelle par méthode",
        "cmp_table": "📋 Tableau comparatif détaillé",
        "cmp_conv": "📉 Convergence comparée — Trapèzes vs Simpson 1/3",
        "cmp_theory": "📐 Comparaison théorique",
        "cmp_method_col": "Critère",
        "cmp_trap_col": "🟠 Trapèzes",
        "cmp_simp_col": "🟣 Simpson 1/3",
        "cmp_simp_wins": "Simpson 1/3 est **{:.1f}×** plus précis que la méthode des Trapèzes",
        "cmp_trap_wins": "Les Trapèzes sont **{:.1f}×** plus précis (cas inhabituel)",
        "cmp_equal": "Les deux méthodes donnent une précision similaire",
        "cmp_no_exact": "⚠️ Valeur exacte non disponible — comparaison relative uniquement",
        "cmp_params_display": "📌 Paramètres de la comparaison",
        "cmp_ratio_label": "Rapport de précision (Trap/Simp)",
        "cmp_order_trap": "O(h²)",
        "cmp_order_simp": "O(h⁴)",
        "cmp_div_trap": "÷4 si n×2",
        "cmp_div_simp": "÷16 si n×2",
        "cmp_type_trap": "Linéaire (trapèzes)",
        "cmp_type_simp": "Parabolique (arcs)",
        "cmp_constraint_trap": "Aucune",
        "cmp_constraint_simp": "n doit être pair",
    },

    "en": {
        "page_title": "Numerical Methods",
        "app_title": "Numerical_Methods",
        "app_subtitle": "Welcome! 👋 &nbsp; Explore <b>Gauss-Jordan</b>, <b>Simpson 1/3</b>, <b>Trapezoid</b> & <b>Comparison</b> — compute, visualize, learn. <em>Enjoy it 🚀</em>",
        "params": "⚙️ Parameters",
        "method": "Method",
        "methods": ["Gauss-Jordan", "Simpson 1/3", "Trapezoid", "Comparison"],
        "function_choice": "Function",
        "predefined": "Predefined",
        "custom": "Custom",
        "select_function": "Choose a predefined function",
        "enter_function": "Enter your function f(x)",
        "function_hint": "Python syntax: x**2, sin(x), exp(x), log(x), sqrt(x) …",
        "lower": "Lower bound (a)",
        "upper": "Upper bound (b)",
        "intervals": "Number of sub-intervals (n)",
        "show_plot": "Show graph",
        "show_table": "Show evaluation table",
        "compare_exact": "Compare with exact value (SymPy)",
        "convergence": "Convergence analysis",
        "calculate": "🚀 Calculate",
        "results": "📊 Results",
        "numerical_result": "Numerical result",
        "exact_value": "Exact value (SymPy)",
        "abs_error": "Absolute error",
        "rel_error": "Relative error",
        "eval_points": "Evaluation points",
        "about_title": "📘 About the method",
        "about_simp": "**Simpson's 1/3 Rule**: Parabolic approximation. Error O(h⁴). Requires n even.",
        "formula_title": "📐 Formula used",
        "formula_simp": r"$\int_a^b f(x)\,dx \approx \frac{h}{3}\left[f(x_0)+4\sum_{\text{odd}}f(x_i)+2\sum_{\text{even}}f(x_i)+f(x_n)\right]$",
        "download_title": "⬇️ Download results",
        "download_txt": "📄 TXT",
        "download_csv": "📊 CSV",
        "download_pdf": "📕 PDF",
        "error_func": "Error in the entered function",
        "error_calc": "Calculation error",
        "error_exact": "Could not compute exact value.",
        "error_n_even": "n must be even for Simpson 1/3.",
        "conv_title": "📉 Convergence: Error vs n",
        "conv_n": "n (sub-intervals)",
        "conv_err": "Absolute error",
        "language": "🌐 Language / Langue / اللغة",
        "n_must_even": "⚠️ n must be even for Simpson 1/3. n was rounded.",
        "exact_na": "N/A (exact value not computed)",
        "simp_intro_title": "📌 Method objective",
        "simp_intro_text": "**Simpson's 1/3 Rule** numerically approximates the definite integral:",
        "simp_intro_sub": "by splitting [a, b] into **n** sub-intervals (n even) and approximating f(x) with parabolas.",
        "simp_params_display": "📌 Calculation parameters",
        "simp_h_label": "Step h",
        "btn_abs_error": "ℹ️ What is absolute error?",
        "btn_rel_error": "ℹ️ What is relative error?",
        "btn_convergence": "ℹ️ What is convergence?",
        "explain_abs_error": """
**Absolute Error — Definition**

$$E_{\\text{abs}} = |I_{\\text{num}} - I_{\\text{exact}}|$$

For Simpson 1/3: $|E| \\leq \\frac{(b-a)^5}{180 n^4} \\max |f^{(4)}(x)|$

| Error | Interpretation |
|-------|---------------|
| < 10⁻⁶ | ✅ Excellent |
| 10⁻⁶–10⁻³ | ✅ Good |
| 10⁻³–10⁻¹ | ⚠️ Moderate |
| > 10⁻¹ | ❌ Poor — increase n |
""",
        "explain_rel_error": """
**Relative Error — Definition**

$$E_{\\text{rel}} = \\frac{|I_{\\text{num}} - I_{\\text{exact}}|}{|I_{\\text{exact}}|}$$

| Error | Interpretation |
|-------|---------------|
| < 10⁻⁸ | ✅ Machine precision |
| < 10⁻⁶ | ✅ Very high |
| < 10⁻³ | ✅ Good (0.1%) |
| > 10⁻² | ⚠️ Poor — increase n |
""",
        "explain_convergence": """
**Convergence — Definition**

$$E \\sim C \\cdot h^4 \\quad \\Rightarrow \\quad \\text{doubling } n \\text{ divides error by } 2^4 = 16$$

Theoretical log-log slope: **−4** for Simpson 1/3.
""",
        "gj_title": "🔢 Gauss-Jordan Method",
        "gj_subtitle": "Solving linear systems Ax = b with full step-by-step analysis",
        "gj_size": "Matrix size (n×n)",
        "gj_matrix_a": "📋 Matrix A (coefficients)",
        "gj_vector_b": "📋 Vector b (right-hand side)",
        "gj_solve": "🚀 Solve the system",
        "gj_augmented": "Initial augmented matrix [A|b]",
        "gj_steps": "📝 Detailed elimination steps",
        "gj_step": "Step",
        "gj_pivot": "Pivot",
        "gj_row_op": "Row operation",
        "gj_matrix_after": "Matrix after operation",
        "gj_rref": "✅ Reduced Row Echelon Form (RREF)",
        "gj_rref_explain": "The **RREF** is the final matrix after Gauss-Jordan. The solution is read directly from the last column.",
        "gj_solution": "🎯 System solution",
        "gj_x": "x",
        "gj_verification": "🔍 Verification: A·x = b",
        "gj_residual": "Residual ‖Ax − b‖",
        "gj_residual_explain": "‖Ax − b‖ measures how well the solution satisfies the system.",
        "gj_det": "Determinant",
        "gj_rank": "Rank",
        "gj_cond": "Cond. κ",
        "gj_cond_explain": """κ = σ_max / σ_min — larger κ means more unstable.

| κ | Interpretation |
|---|---------------|
| < 100 | ✅ Well-conditioned |
| 100–10⁶ | ⚠️ Moderate |
| ≥ 10⁶ | ❌ Ill-conditioned |
""",
        "gj_unique": "Unique solution",
        "gj_no_solution": "❌ Inconsistent system (no solution)",
        "gj_inf_solutions": "♾️ Infinitely many solutions",
        "gj_singular": "⚠️ Singular matrix (det = 0)",
        "gj_about": "📘 About Gauss-Jordan",
        "gj_about_text": """**Gauss-Jordan** reduces the augmented matrix to its **RREF**.

**Algorithm:** Form [A|b] → partial pivot → normalize → eliminate → read solution.

**Complexity:** O(n³)""",
        "gj_formula": "General form",
        "gj_col": "Column",
        "gj_swap": "Row swap",
        "gj_normalize": "Pivot normalization",
        "gj_eliminate": "Elimination",
        "gj_summary": "📊 Results summary",
        "gj_ax": "A·xᵢ",
        "gj_bi": "bᵢ",
        "gj_diff": "|A·xᵢ − bᵢ|",
        "gj_enter_a": "Enter the coefficients of A",
        "gj_enter_b": "Enter the values of b",
        "gj_row": "Row",
        "nav_gj": "🔢 Gauss-Jordan",
        "nav_simp": "∫ Simpson 1/3",
        "nav_trap": "∫∫ Trapezoid",
        "nav_comp": "⚖️ Comparison",
        "page_label": "Active page",
        "gj_residual_negligible": "✅ Negligible residual — exact solution",
        "gj_residual_low": "ℹ️ Low residual — good precision",
        "gj_residual_high": "⚠️ Notable residual — check conditioning",
        "gj_what_is_cond": "ℹ️ What is the condition number?",
        "gj_what_is_rref": "ℹ️ What is RREF?",
        "gj_what_is_residual": "ℹ️ What is the residual?",
        "gj_solution_direct": "🎯 Solution — x",
        "trap_title": "∫∫ Generalized Trapezoidal Rule",
        "trap_subtitle": "Numerical integration via linear approximation — O(h²) — full analysis",
        "trap_intro_title": "📌 Method objective",
        "trap_intro_text": "The **Trapezoidal Rule** numerically approximates the definite integral:",
        "trap_intro_sub": "by splitting [a, b] into **n** sub-intervals and approximating f(x) with straight-line segments (trapezoids).",
        "trap_params_display": "📌 Calculation parameters",
        "trap_h_label": "Step h",
        "trap_about": "📘 About the Trapezoidal Rule",
        "trap_about_text": """**Trapezoidal Rule**: piecewise linear approximation. Error order O(h²).

**Formula:** $\\frac{h}{2}[f(x_0) + 2f(x_1) + \\cdots + 2f(x_{n-1}) + f(x_n)]$

**Error bound:** $|E| \\leq \\frac{(b-a)^3}{12n^2} \\max|f''(x)|$

**Complexity:** O(n) — simple to implement, O(h²) convergence.""",
        "trap_formula_title": "📐 Trapezoidal Formula",
        "about_trap": "**Trapezoidal Rule**: Linear approximation. Error O(h²). No constraint on n.",
        "btn_trap_error": "ℹ️ What is the Trapezoid error?",
        "explain_trap_error": """
**Trapezoidal Rule Error**

$$|E_T| \\leq \\frac{(b-a)^3}{12n^2} \\max_{[a,b]} |f''(x)|$$

Doubling n divides error by **2² = 4**.

| Order | Method | Reduction when n×2 |
|-------|--------|-------------------|
| O(h²) | Trapezoid | ÷4 |
| O(h⁴) | Simpson 1/3 | ÷16 |
""",
        "cmp_title": "⚖️ Comparison: Trapezoid vs Simpson 1/3",
        "cmp_subtitle": "In-depth comparative analysis — precision, convergence, side-by-side visualization",
        "cmp_calculate": "🚀 Run comparison",
        "cmp_results": "📊 Comparative results",
        "cmp_exact": "Exact value",
        "cmp_trap_res": "Trapezoid result",
        "cmp_simp_res": "Simpson result",
        "cmp_trap_err": "Trapezoid error",
        "cmp_simp_err": "Simpson error",
        "cmp_winner": "🏆 Precision verdict",
        "cmp_graph": "📊 Comparative visualization — Both methods overlaid",
        "cmp_graph_side": "🔍 Individual method visualization",
        "cmp_table": "📋 Detailed comparison table",
        "cmp_conv": "📉 Convergence comparison — Trapezoid vs Simpson 1/3",
        "cmp_theory": "📐 Theoretical comparison",
        "cmp_method_col": "Criterion",
        "cmp_trap_col": "🟠 Trapezoid",
        "cmp_simp_col": "🟣 Simpson 1/3",
        "cmp_simp_wins": "Simpson 1/3 is **{:.1f}×** more precise than the Trapezoidal Rule",
        "cmp_trap_wins": "Trapezoidal Rule is **{:.1f}×** more precise (unusual case)",
        "cmp_equal": "Both methods give similar precision",
        "cmp_no_exact": "⚠️ Exact value unavailable — relative comparison only",
        "cmp_params_display": "📌 Comparison parameters",
        "cmp_ratio_label": "Precision ratio (Trap/Simp)",
        "cmp_order_trap": "O(h²)",
        "cmp_order_simp": "O(h⁴)",
        "cmp_div_trap": "÷4 when n×2",
        "cmp_div_simp": "÷16 when n×2",
        "cmp_type_trap": "Linear (trapezoids)",
        "cmp_type_simp": "Parabolic (arcs)",
        "cmp_constraint_trap": "None",
        "cmp_constraint_simp": "n must be even",
    },

    "ar": {
        "page_title": "الطرق العددية",
        "app_title": "الطرق_العددية",
        "app_subtitle": "أهلاً وسهلاً! 👋 &nbsp; استكشف <b>غاوس-جوردان</b>، <b>سيمبسون 1/3</b>، <b>شبه المنحرف</b> و<b>المقارنة</b> — احسب، تصور، تعلّم. <em>Enjoy it 🚀</em>",
        "params": "⚙️ الإعدادات",
        "method": "الطريقة",
        "methods": ["غاوس-جوردان", "سيمبسون 1/3", "شبه المنحرف", "مقارنة"],
        "function_choice": "الدالة",
        "predefined": "دالة جاهزة",
        "custom": "دالة مخصصة",
        "select_function": "اختر دالة جاهزة",
        "enter_function": "أدخل الدالة f(x)",
        "function_hint": "صيغة Python: x**2, sin(x), exp(x), log(x), sqrt(x) …",
        "lower": "الحد الأدنى (a)",
        "upper": "الحد الأعلى (b)",
        "intervals": "عدد الأجزاء الفرعية (n)",
        "show_plot": "عرض الرسم البياني",
        "show_table": "عرض جدول النقاط",
        "compare_exact": "مقارنة مع القيمة الدقيقة (SymPy)",
        "convergence": "تحليل التقارب",
        "calculate": "🚀 حساب",
        "results": "📊 النتائج",
        "numerical_result": "النتيجة العددية",
        "exact_value": "القيمة الدقيقة (SymPy)",
        "abs_error": "الخطأ المطلق",
        "rel_error": "الخطأ النسبي",
        "eval_points": "نقاط التقييم",
        "about_title": "📘 حول الطريقة",
        "about_simp": "**سيمبسون 1/3**: تقريب مكافئ. الخطأ O(h⁴). يتطلب n زوجياً.",
        "formula_title": "📐 الصيغة المستخدمة",
        "formula_simp": r"$\int_a^b f(x)\,dx \approx \frac{h}{3}\left[f(x_0)+4\sum f(x_{\text{فردي}})+2\sum f(x_{\text{زوجي}})+f(x_n)\right]$",
        "download_title": "⬇️ تحميل النتائج",
        "download_txt": "📄 TXT",
        "download_csv": "📊 CSV",
        "download_pdf": "📕 PDF",
        "error_func": "خطأ في الدالة المدخلة",
        "error_calc": "خطأ في الحساب",
        "error_exact": "تعذّر حساب القيمة الدقيقة.",
        "error_n_even": "يجب أن يكون n زوجياً لـ Simpson 1/3.",
        "conv_title": "📉 التقارب: الخطأ مقابل n",
        "conv_n": "n (الأجزاء الفرعية)",
        "conv_err": "الخطأ المطلق",
        "language": "🌐 اللغة / Langue / Language",
        "n_must_even": "⚠️ يجب أن يكون n زوجياً لسيمبسون. تم تقريب n.",
        "exact_na": "غير متاح",
        "simp_intro_title": "📌 هدف الطريقة",
        "simp_intro_text": "طريقة **سيمبسون 1/3** تُقدّر عددياً التكامل المحدد:",
        "simp_intro_sub": "بتقسيم [a, b] إلى **n** جزء فرعي (n زوجي) وتقريب f(x) بمكافئات.",
        "simp_params_display": "📌 معاملات الحساب",
        "simp_h_label": "الخطوة h",
        "btn_abs_error": "ℹ️ ما هو الخطأ المطلق؟",
        "btn_rel_error": "ℹ️ ما هو الخطأ النسبي؟",
        "btn_convergence": "ℹ️ ما هو التقارب؟",
        "explain_abs_error": "$$E_{\\text{abs}} = |I_{\\text{num}} - I_{\\text{exact}}|$$ لسيمبسون: $|E| \\leq \\frac{(b-a)^5}{180 n^4} \\max |f^{(4)}(x)|$",
        "explain_rel_error": "$$E_{\\text{rel}} = \\frac{|I_{\\text{num}} - I_{\\text{exact}}|}{|I_{\\text{exact}}|}$$ بلا أبعاد ويسمح بالمقارنة.",
        "explain_convergence": "$$E \\sim C \\cdot h^4 \\Rightarrow \\text{مضاعفة } n \\text{ تقسّم الخطأ على } 16$$",
        "gj_title": "🔢 طريقة غاوس-جوردان",
        "gj_subtitle": "حل الأنظمة الخطية Ax = b مع تحليل كامل خطوة بخطوة",
        "gj_size": "حجم المصفوفة (n×n)",
        "gj_matrix_a": "📋 المصفوفة A (المعاملات)",
        "gj_vector_b": "📋 المتجه b (الطرف الأيمن)",
        "gj_solve": "🚀 حل النظام",
        "gj_augmented": "المصفوفة المعززة الأولية [A|b]",
        "gj_steps": "📝 خطوات الحذف التفصيلية",
        "gj_step": "خطوة",
        "gj_pivot": "المحور",
        "gj_row_op": "عملية الصف",
        "gj_matrix_after": "المصفوفة بعد العملية",
        "gj_rref": "✅ الصورة المختزلة للدرج (RREF)",
        "gj_rref_explain": "الصورة النهائية للمصفوفة بعد حذف غاوس-جوردان. الحل يُقرأ مباشرة من العمود الأخير.",
        "gj_solution": "🎯 حل النظام",
        "gj_x": "x",
        "gj_verification": "🔍 التحقق: A·x = b",
        "gj_residual": "البقايا ‖Ax − b‖",
        "gj_residual_explain": "‖Ax − b‖ يقيس مدى دقة الحل المحسوب.",
        "gj_det": "محدد A",
        "gj_rank": "الرتبة",
        "gj_cond": "رقم الشرط κ",
        "gj_cond_explain": "κ = σ_max / σ_min — كلما كان أكبر زاد عدم الاستقرار.",
        "gj_unique": "حل وحيد",
        "gj_no_solution": "❌ نظام غير متوافق (لا يوجد حل)",
        "gj_inf_solutions": "♾️ حلول لا نهائية",
        "gj_singular": "⚠️ مصفوفة شاذة (det = 0)",
        "gj_about": "📘 حول طريقة غاوس-جوردان",
        "gj_about_text": "**غاوس-جوردان** تختزل المصفوفة المعززة إلى RREF. التعقيد: O(n³)",
        "gj_formula": "الصيغة العامة",
        "gj_col": "العمود",
        "gj_swap": "تبديل الصفوف",
        "gj_normalize": "تطبيع المحور",
        "gj_eliminate": "الحذف",
        "gj_summary": "📊 ملخص النتائج",
        "gj_ax": "A·xᵢ",
        "gj_bi": "bᵢ",
        "gj_diff": "|A·xᵢ − bᵢ|",
        "gj_enter_a": "أدخل معاملات A",
        "gj_enter_b": "أدخل قيم b",
        "gj_row": "صف",
        "nav_gj": "🔢 غاوس-جوردان",
        "nav_simp": "∫ سيمبسون 1/3",
        "nav_trap": "∫∫ شبه المنحرف",
        "nav_comp": "⚖️ مقارنة",
        "page_label": "الصفحة النشطة",
        "gj_residual_negligible": "✅ بقايا ضئيلة — حل دقيق",
        "gj_residual_low": "ℹ️ بقايا منخفضة — دقة جيدة",
        "gj_residual_high": "⚠️ بقايا ملحوظة — تحقق من التكييف",
        "gj_what_is_cond": "ℹ️ ما هو رقم الشرط؟",
        "gj_what_is_rref": "ℹ️ ما هي RREF؟",
        "gj_what_is_residual": "ℹ️ ما معنى البقايا؟",
        "gj_solution_direct": "🎯 الحل — x",
        "trap_title": "∫∫ طريقة شبه المنحرف المعممة",
        "trap_subtitle": "تكامل عددي بتقريب خطي — O(h²) — تحليل كامل",
        "trap_intro_title": "📌 هدف الطريقة",
        "trap_intro_text": "طريقة **شبه المنحرف** تُقدّر عددياً التكامل المحدد:",
        "trap_intro_sub": "بتقسيم [a, b] إلى **n** جزء فرعي وتقريب f(x) بأجزاء خطية (أشباه منحرف).",
        "trap_params_display": "📌 معاملات الحساب",
        "trap_h_label": "الخطوة h",
        "trap_about": "📘 حول طريقة شبه المنحرف",
        "trap_about_text": "**طريقة شبه المنحرف**: تقريب خطي بالقطعة. الخطأ O(h²). لا قيد على n.",
        "trap_formula_title": "📐 صيغة شبه المنحرف",
        "about_trap": "**شبه المنحرف**: تقريب خطي. الخطأ O(h²). لا قيد على n.",
        "btn_trap_error": "ℹ️ ما هو خطأ شبه المنحرف؟",
        "explain_trap_error": "$$|E_T| \\leq \\frac{(b-a)^3}{12n^2} \\max |f''(x)|$$ مضاعفة n تقسّم الخطأ على **4**.",
        "cmp_title": "⚖️ مقارنة: شبه المنحرف vs سيمبسون 1/3",
        "cmp_subtitle": "تحليل مقارن شامل — الدقة، التقارب، التمثيل البياني جنباً إلى جنب",
        "cmp_calculate": "🚀 تشغيل المقارنة",
        "cmp_results": "📊 النتائج المقارنة",
        "cmp_exact": "القيمة الدقيقة",
        "cmp_trap_res": "نتيجة شبه المنحرف",
        "cmp_simp_res": "نتيجة سيمبسون",
        "cmp_trap_err": "خطأ شبه المنحرف",
        "cmp_simp_err": "خطأ سيمبسون",
        "cmp_winner": "🏆 حكم الدقة",
        "cmp_graph": "📊 التمثيل البياني المقارن — الطريقتان معاً",
        "cmp_graph_side": "🔍 التمثيل الفردي لكل طريقة",
        "cmp_table": "📋 جدول المقارنة التفصيلي",
        "cmp_conv": "📉 مقارنة التقارب — شبه المنحرف vs سيمبسون",
        "cmp_theory": "📐 المقارنة النظرية",
        "cmp_method_col": "المعيار",
        "cmp_trap_col": "🟠 شبه المنحرف",
        "cmp_simp_col": "🟣 سيمبسون 1/3",
        "cmp_simp_wins": "سيمبسون 1/3 أدق بـ **{:.1f}×** من طريقة شبه المنحرف",
        "cmp_trap_wins": "شبه المنحرف أدق بـ **{:.1f}×** (حالة غير معتادة)",
        "cmp_equal": "الطريقتان تعطيان دقة متماثلة",
        "cmp_no_exact": "⚠️ القيمة الدقيقة غير متاحة — مقارنة نسبية فقط",
        "cmp_params_display": "📌 معاملات المقارنة",
        "cmp_ratio_label": "نسبة الدقة (شبه منحرف/سيمبسون)",
        "cmp_order_trap": "O(h²)",
        "cmp_order_simp": "O(h⁴)",
        "cmp_div_trap": "÷4 عند n×2",
        "cmp_div_simp": "÷16 عند n×2",
        "cmp_type_trap": "خطي (أشباه منحرف)",
        "cmp_type_simp": "مكافئ (أقواس)",
        "cmp_constraint_trap": "لا يوجد",
        "cmp_constraint_simp": "n يجب أن يكون زوجياً",
    }
}

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Méthodes Numériques",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #070d1a;
}
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.terminal-wrap { padding: 2rem 0 0.5rem 0; }
.terminal-prefix {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem; color: #10b981;
    letter-spacing: 0.05em; margin-bottom: 0.15rem;
}
.terminal-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(90deg, #06b6d4 0%, #10b981 60%, #a78bfa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -0.03em; line-height: 1.1; margin: 0;
}
.terminal-cursor {
    display: inline-block; width: 3px; height: 2.2rem;
    background: #06b6d4; margin-left: 4px; vertical-align: middle;
    animation: blink-cursor 1s step-end infinite;
}
@keyframes blink-cursor { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
.terminal-subtitle {
    font-family: 'Inter', sans-serif; font-size: 1.0rem;
    color: #94a3b8; margin-top: 0.6rem; letter-spacing: 0.01em;
}
.terminal-subtitle b { color: #06b6d4; }

.stButton > button {
    background: linear-gradient(135deg, #0a1628 0%, #0f2040 100%);
    color: #cbd5e1; border: 1px solid #1e3a5f; border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace; font-weight: 600;
    font-size: 0.88rem; letter-spacing: 0.03em;
    padding: 0.55em 1.3em; transition: all 0.22s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
    color: #fff; border-color: #06b6d4; transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(6,182,212,0.35);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
    color: #fff; border-color: #06b6d4;
    box-shadow: 0 4px 14px rgba(6,182,212,0.3);
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%);
    box-shadow: 0 6px 22px rgba(6,182,212,0.5);
}

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0a1628 0%, #0f2040 100%);
    border: 1px solid #1e3a5f; border-radius: 10px; padding: 0.9rem 1.1rem;
}
[data-testid="metric-container"]:hover {
    border-color: #06b6d4; box-shadow: 0 0 14px rgba(6,182,212,0.15);
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    color: #64748b !important; text-transform: uppercase; letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    color: #06b6d4 !important; font-size: 1.3rem !important;
}

div[data-testid="stExpander"] {
    border: 1px solid #1e3a5f; border-radius: 10px;
    background: #0a1628; transition: border-color 0.2s;
}
div[data-testid="stExpander"]:hover { border-color: #06b6d466; }
div[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.88rem; color: #94a3b8;
}

.info-banner {
    background: linear-gradient(135deg, #0a1e38 0%, #0c2340 100%);
    border: 1px solid #06b6d433; border-left: 4px solid #06b6d4;
    border-radius: 10px; padding: 1.1rem 1.4rem; margin: 0.8rem 0 1.2rem 0;
}
.info-banner-purple {
    background: linear-gradient(135deg, #1a0a38 0%, #240c40 100%);
    border: 1px solid #a78bfa33; border-left: 4px solid #a78bfa;
    border-radius: 10px; padding: 1.1rem 1.4rem; margin: 0.8rem 0 1.2rem 0;
}
.info-banner-orange {
    background: linear-gradient(135deg, #381a0a 0%, #402408 100%);
    border: 1px solid #f59e0b33; border-left: 4px solid #f59e0b;
    border-radius: 10px; padding: 1.1rem 1.4rem; margin: 0.8rem 0 1.2rem 0;
}
.winner-banner {
    background: linear-gradient(135deg, #051a12 0%, #082010 100%);
    border: 2px solid #10b981; border-radius: 14px;
    padding: 1.5rem 2rem; margin: 1rem 0;
    box-shadow: 0 0 30px rgba(16,185,129,0.2);
    text-align: center;
}
.solution-card {
    background: linear-gradient(135deg, #051a12, #0a1628);
    border: 1px solid #10b981; border-radius: 12px;
    padding: 1.4rem 1.6rem; margin: 1rem 0;
    box-shadow: 0 0 20px rgba(16,185,129,0.12);
}
.solution-card h3 {
    color: #10b981; font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem; margin-bottom: 0.8rem; letter-spacing: 0.05em;
}
.divider-accent {
    height: 1px;
    background: linear-gradient(90deg, transparent, #06b6d4, transparent);
    margin: 1.5rem 0; border: none;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070d1a 0%, #0a1628 100%);
    border-right: 1px solid #1e3a5f;
}
.stSuccess { border-left-color: #10b981 !important; }
.stWarning { border-left-color: #f59e0b !important; }
.stError   { border-left-color: #ef4444 !important; }
.stInfo    { border-left-color: #06b6d4 !important; }
[data-testid="stDataFrame"] { border: 1px solid #1e3a5f; border-radius: 8px; }
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "gj"

# ─────────────────────────────────────────────
#  LANGUAGE SELECTOR
# ─────────────────────────────────────────────
with st.sidebar:
    lang_choice = st.selectbox(
        "🌐 Langue / Language / اللغة",
        options=["fr", "en", "ar"],
        format_func=lambda x: {"fr": "🇫🇷 Français", "en": "🇬🇧 English", "ar": "🇩🇿 العربية"}[x],
    )
T = LANG[lang_choice]

if lang_choice == "ar":
    st.markdown('<style>body, .stMarkdown, .stText { direction: rtl; text-align: right; }</style>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="terminal-wrap">
    <div class="terminal-prefix">&lt;/&gt; Python Programming &nbsp;·&nbsp; ZMHY</div>
    <div class="terminal-title">{T["app_title"]}<span class="terminal-cursor"></span></div>
    <div class="terminal-subtitle">{T["app_subtitle"]}</div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  NAVIGATION — 4 buttons
# ─────────────────────────────────────────────
nc1, nc2, nc3, nc4 = st.columns([2, 2, 2, 2])
with nc1:
    if st.button(T["nav_gj"], use_container_width=True,
                 type="primary" if st.session_state["page"] == "gj" else "secondary"):
        st.session_state["page"] = "gj"; st.rerun()
with nc2:
    if st.button(T["nav_simp"], use_container_width=True,
                 type="primary" if st.session_state["page"] == "simp" else "secondary"):
        st.session_state["page"] = "simp"; st.rerun()
with nc3:
    if st.button(T["nav_trap"], use_container_width=True,
                 type="primary" if st.session_state["page"] == "trap" else "secondary"):
        st.session_state["page"] = "trap"; st.rerun()
with nc4:
    if st.button(T["nav_comp"], use_container_width=True,
                 type="primary" if st.session_state["page"] == "comp" else "secondary"):
        st.session_state["page"] = "comp"; st.rerun()

st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  MATH HELPERS — GAUSS-JORDAN
# ═══════════════════════════════════════════════════════════════════════
def fmt_matrix(M, highlight_col=None, n_cols=None):
    rows, cols = M.shape
    sep_col = n_cols
    html = '<table style="border-collapse:collapse;font-family:IBM Plex Mono,monospace;font-size:0.82rem;margin:0.4rem 0;">'
    for i in range(rows):
        html += "<tr>"
        for j in range(cols):
            border_left = "border-left:2px solid #06b6d4;" if (sep_col is not None and j == sep_col) else ""
            bg = "background:#06b6d415;" if (highlight_col is not None and j == highlight_col) else ""
            val = M[i, j]; rounded = round(val, 6)
            if abs(rounded) < 1e-9: txt = "0"
            elif rounded == int(rounded) and abs(rounded) < 1e9: txt = f"{int(rounded):+d}"
            else:
                s = f"{rounded:+.5f}".rstrip('0')
                if s.endswith('.') or s.endswith('+') or s.endswith('-'): s += '0'
                txt = s
            html += (f'<td style="padding:4px 12px;{border_left}{bg}color:#e2e8f0;'
                     f'min-width:70px;text-align:right;border-bottom:1px solid #1e3a5f;">{txt}</td>')
        html += "</tr>"
    html += "</table>"
    return html


def gauss_jordan_detailed(A_orig, b_orig):
    n = len(b_orig)
    Aug = np.hstack([A_orig.copy().astype(float), b_orig.copy().reshape(-1, 1).astype(float)])
    steps = []; det_sign = 1; pivot_row = 0
    for col in range(n):
        max_row = pivot_row + np.argmax(np.abs(Aug[pivot_row:, col]))
        if abs(Aug[max_row, col]) < 1e-12: continue
        if max_row != pivot_row:
            Aug[[pivot_row, max_row]] = Aug[[max_row, pivot_row]]; det_sign *= -1
            steps.append({"type": "swap", "col": col, "pivot_row": pivot_row,
                           "desc": f"R{pivot_row+1} ↔ R{max_row+1}", "matrix": Aug.copy()})
        pivot_val = Aug[pivot_row, col]
        Aug[pivot_row] = Aug[pivot_row] / pivot_val
        steps.append({"type": "normalize", "col": col, "pivot_row": pivot_row,
                       "desc": f"R{pivot_row+1} ← R{pivot_row+1} / ({smart_fmt(pivot_val)})",
                       "matrix": Aug.copy(), "pivot_val": pivot_val})
        for row in range(n):
            if row == pivot_row: continue
            factor = Aug[row, col]
            if abs(factor) < 1e-12: continue
            Aug[row] = Aug[row] - factor * Aug[pivot_row]
            steps.append({"type": "eliminate", "col": col, "target_row": row, "pivot_row": pivot_row,
                           "desc": f"R{row+1} ← R{row+1} − ({smart_fmt(factor)})·R{pivot_row+1}",
                           "matrix": Aug.copy(), "factor": factor})
        pivot_row += 1
    rref = Aug.copy()
    rank = sum(1 for i in range(n) if np.any(np.abs(rref[i, :n]) > 1e-10))
    try: det = float(np.linalg.det(A_orig.astype(float)))
    except: det = None
    try: cond = float(np.linalg.cond(A_orig.astype(float)))
    except: cond = None
    status = "unique"
    if rank < n:
        for i in range(rank, n):
            if abs(rref[i, n]) > 1e-10:
                return None, steps, rref, det, rank, cond, "no_solution"
        return None, steps, rref, det, rank, cond, "infinite"
    solution = rref[:, n].copy()
    return solution, steps, rref, det, rank, cond, status


# ═══════════════════════════════════════════════════════════════════════
#  MATH HELPERS — INTEGRATION
# ═══════════════════════════════════════════════════════════════════════
_SAFE_NAMES = {
    'pi': np.pi, 'e': np.e,
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
    'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
    'sinh': np.sinh, 'sh': np.sinh, 'cosh': np.cosh, 'ch': np.cosh,
    'tanh': np.tanh, 'th': np.tanh, 'asinh': np.arcsinh, 'argsh': np.arcsinh,
    'acosh': np.arccosh, 'argch': np.arccosh, 'atanh': np.arctanh, 'argth': np.arctanh,
    'exp': np.exp, 'log': np.log, 'ln': np.log,
    'log10': np.log10, 'sqrt': np.sqrt, 'abs': np.abs,
    'floor': np.floor, 'ceil': np.ceil, 'sign': np.sign,
}

def evaluate(func_str, x):
    ns = dict(_SAFE_NAMES); ns['x'] = x
    return eval(func_str, {"__builtins__": {}}, ns)

def exact_integral(func_str, a, b):
    x = sp.Symbol('x')
    try:
        custom_locals = {
            'sh': sp.sinh, 'ch': sp.cosh, 'th': sp.tanh,
            'argsh': sp.asinh, 'argch': sp.acosh, 'argth': sp.atanh,
            'ln': sp.log, 'arctan': sp.atan, 'atan': sp.atan,
            'arcsin': sp.asin, 'arccos': sp.acos
        }
        expr = sp.sympify(func_str, locals=custom_locals)
        val = sp.integrate(expr, (x, a, b))
        return float(val.evalf())
    except: return None

def simpson_13(func_str, a, b, n):
    if n % 2 != 0: n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = evaluate(func_str, x)
    result = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
    return x, y, result, n

def trapezoid_rule(func_str, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = evaluate(func_str, x)
    result = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return x, y, result

def convergence_data(func_str, a, b, exact):
    ns = list(range(2, 52, 2)); errors = []
    for ni in ns:
        try: _, _, res, _ = simpson_13(func_str, a, b, ni); errors.append(abs(res - exact))
        except: errors.append(np.nan)
    return ns, errors

def convergence_data_trap(func_str, a, b, exact):
    ns = list(range(2, 102, 2)); errors = []
    for ni in ns:
        try: _, _, res = trapezoid_rule(func_str, a, b, ni); errors.append(abs(res - exact))
        except: errors.append(np.nan)
    return ns, errors


# ═══════════════════════════════════════════════════════════════════════
#  DOWNLOAD HELPERS
# ═══════════════════════════════════════════════════════════════════════
def make_txt_simp(func_str, a, b, n, result, exact_val, x_vals, y_vals):
    buf = StringIO(); now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    buf.write("=" * 52 + "\n"); buf.write(f"  RAPPORT SIMPSON 1/3 — {now}\n"); buf.write("=" * 52 + "\n\n")
    buf.write(f"  Fonction        : f(x) = {func_str}\n")
    buf.write(f"  Intervalle      : [{a}, {b}]\n")
    buf.write(f"  Sous-intervalles: n = {n}\n")
    buf.write(f"  Pas h           : {(b-a)/n:.8f}\n\n")
    buf.write(f"  Résultat num.   : {result:.12f}\n")
    if exact_val is not None:
        buf.write(f"  Valeur exacte   : {exact_val:.12f}\n")
        buf.write(f"  Erreur absolue  : {abs(result - exact_val):.4e}\n")
        buf.write(f"  Erreur relative : {abs(result - exact_val)/abs(exact_val):.4e}\n")
    if x_vals is not None:
        buf.write("\n  Points d'évaluation:\n")
        buf.write(f"  {'i':>4}  {'x_i':>14}  {'f(x_i)':>14}\n")
        buf.write("  " + "-" * 36 + "\n")
        for i, (xi, yi) in enumerate(zip(x_vals, y_vals)):
            buf.write(f"  {i:>4}  {xi:>14.8f}  {yi:>14.8f}\n")
    buf.write("\n" + "=" * 52 + "\n")
    return buf.getvalue().encode("utf-8")

def make_csv_simp(x_vals, y_vals):
    if x_vals is None: return b""
    return pd.DataFrame({"x_i": x_vals, "f(x_i)": y_vals}).to_csv(index=True).encode("utf-8")

def make_txt_trap(func_str, a, b, n, result, exact_val, x_vals, y_vals):
    buf = StringIO(); now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    buf.write("=" * 52 + "\n"); buf.write(f"  RAPPORT TRAPÈZES — {now}\n"); buf.write("=" * 52 + "\n\n")
    buf.write(f"  Fonction        : f(x) = {func_str}\n")
    buf.write(f"  Intervalle      : [{a}, {b}]\n")
    buf.write(f"  Sous-intervalles: n = {n}\n")
    buf.write(f"  Pas h           : {(b-a)/n:.8f}\n\n")
    buf.write(f"  Résultat num.   : {result:.12f}\n")
    if exact_val is not None:
        buf.write(f"  Valeur exacte   : {exact_val:.12f}\n")
        buf.write(f"  Erreur absolue  : {abs(result - exact_val):.4e}\n")
        buf.write(f"  Erreur relative : {abs(result - exact_val)/abs(exact_val):.4e}\n")
    if x_vals is not None:
        buf.write("\n  Points d'évaluation:\n")
        buf.write(f"  {'i':>4}  {'x_i':>14}  {'f(x_i)':>14}\n")
        buf.write("  " + "-" * 36 + "\n")
        for i, (xi, yi) in enumerate(zip(x_vals, y_vals)):
            buf.write(f"  {i:>4}  {xi:>14.8f}  {yi:>14.8f}\n")
    buf.write("\n" + "=" * 52 + "\n")
    return buf.getvalue().encode("utf-8")

def make_csv_trap(x_vals, y_vals):
    if x_vals is None: return b""
    return pd.DataFrame({"x_i": x_vals, "f(x_i)": y_vals}).to_csv(index=True).encode("utf-8")

def make_txt_cmp(func_str, a, b, n, trap_res, simp_res, exact_val):
    buf = StringIO(); now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    buf.write("=" * 60 + "\n")
    buf.write(f"  RAPPORT DE COMPARAISON — {now}\n")
    buf.write("  Trapèzes vs Simpson 1/3\n")
    buf.write("=" * 60 + "\n\n")
    buf.write(f"  Fonction   : f(x) = {func_str}\n")
    buf.write(f"  Intervalle : [{a}, {b}]\n")
    buf.write(f"  n          : {n}\n\n")
    buf.write(f"  Résultat Trapèzes  : {trap_res:.12f}\n")
    buf.write(f"  Résultat Simpson   : {simp_res:.12f}\n")
    if exact_val is not None:
        buf.write(f"  Valeur exacte      : {exact_val:.12f}\n\n")
        te = abs(trap_res - exact_val); se = abs(simp_res - exact_val)
        buf.write(f"  Erreur Trapèzes    : {te:.4e}\n")
        buf.write(f"  Erreur Simpson     : {se:.4e}\n")
        if se > 0: buf.write(f"  Rapport Trap/Simp  : {te/se:.2f}×\n")
        buf.write(f"  Méthode gagnante   : {'Simpson 1/3' if se < te else 'Trapèzes'}\n")
    buf.write("\n  COMPARAISON THÉORIQUE\n")
    buf.write("  " + "-" * 40 + "\n")
    buf.write(f"  Ordre Trapèzes  : O(h²) — réduction ÷4 si n×2\n")
    buf.write(f"  Ordre Simpson   : O(h⁴) — réduction ÷16 si n×2\n")
    buf.write("\n" + "=" * 60 + "\n")
    return buf.getvalue().encode("utf-8")

def make_csv_cmp(func_str, a, b, n, trap_res, simp_res, exact_val, x_trap, y_trap, x_simp, y_simp):
    rows = []
    for i in range(len(x_trap)):
        rows.append({
            "i": i,
            "x_trap": x_trap[i],
            "f_trap": y_trap[i],
            "x_simp": x_simp[i] if i < len(x_simp) else "",
            "f_simp": y_simp[i] if i < len(y_simp) else "",
        })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

def make_txt_gj(n_size, A, b_vec, solution, steps, det, rank, cond):
    buf = StringIO(); now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    buf.write("=" * 60 + "\n"); buf.write(f"  RAPPORT GAUSS-JORDAN — {now}\n"); buf.write("=" * 60 + "\n\n")
    buf.write(f"  Taille du système: {n_size}×{n_size}\n\n")
    buf.write("  Matrice A:\n")
    for row in A: buf.write("  " + "  ".join(f"{v:>10.5f}" for v in row) + "\n")
    buf.write("\n  Vecteur b:\n")
    buf.write("  " + "  ".join(f"{v:>10.5f}" for v in b_vec) + "\n\n")
    if det is not None: buf.write(f"  Déterminant     : {smart_fmt(det)}\n")
    buf.write(f"  Rang            : {rank}\n")
    if cond is not None: buf.write(f"  Conditionnement : {cond:.4e}\n")
    buf.write(f"  Nombre d'étapes : {len(steps)}\n\n")
    if solution is not None:
        buf.write("  Solution:\n")
        for i, xi in enumerate(solution): buf.write(f"    x{i+1} = {smart_fmt(xi)}\n")
        buf.write("\n  Vérification A·x:\n"); ax = A @ solution
        for i in range(n_size):
            diff = abs(ax[i] - b_vec[i])
            buf.write(f"    A·x[{i+1}] = {smart_fmt(ax[i])}  b[{i+1}] = {smart_fmt(b_vec[i])}  err = {diff:.2e}\n")
    buf.write("\n" + "=" * 60 + "\n")
    return buf.getvalue().encode("utf-8")

def make_csv_gj(solution, A, b_vec):
    rows = []
    if solution is not None:
        ax = A @ solution
        for i, xi in enumerate(solution):
            rows.append({"variable": f"x{i+1}", "valeur": xi, "A·x_i": ax[i],
                         "b_i": b_vec[i], "résidu": abs(ax[i] - b_vec[i])})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

def make_pdf_generic(title_str, content_lines, fig=None):
    """Generic PDF maker."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    pdf = FPDF(); pdf.add_page(); pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_fill_color(7, 13, 26); pdf.set_text_color(6, 182, 212)
    pdf.rect(0, 0, 210, 28, 'F'); pdf.set_xy(10, 8); pdf.cell(0, 12, title_str, ln=True)
    pdf.set_text_color(30, 30, 30); pdf.set_font("Helvetica", "", 10); pdf.set_xy(10, 32)
    pdf.cell(0, 6, f"Date : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)
    for label, val in content_lines:
        pdf.cell(60, 7, f"  {label}", border="B")
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, str(val), border="B", ln=True)
        pdf.set_font("Helvetica", "", 10)
    if fig is not None:
        img_buf = BytesIO(); fig.savefig(img_buf, format="png", dpi=110, bbox_inches="tight")
        img_buf.seek(0); pdf.add_page()
        pdf.set_font("Helvetica", "B", 12); pdf.set_fill_color(10, 22, 40)
        pdf.set_text_color(6, 182, 212); pdf.cell(0, 8, "  Graphique", ln=True, fill=True)
        pdf.image(img_buf, x=10, w=190)
    return bytes(pdf.output())

def make_pdf_simp(func_str, a, b, n, result, exact_val, fig=None):
    lines = [
        ("Fonction", f"f(x) = {func_str}"),
        ("Intervalle", f"[{a}, {b}]"),
        ("n", str(n)), ("h = (b-a)/n", f"{(b-a)/n:.8f}"),
        ("Résultat numérique", f"{result:.12f}"),
    ]
    if exact_val is not None:
        abs_err = abs(result - exact_val)
        rel_err = abs_err / abs(exact_val) if exact_val != 0 else float("inf")
        lines += [("Valeur exacte", f"{exact_val:.12f}"),
                  ("Erreur absolue", f"{abs_err:.4e}"),
                  ("Erreur relative", f"{rel_err:.4e}")]
    return make_pdf_generic("Rapport Simpson 1/3", lines, fig)

def make_pdf_trap(func_str, a, b, n, result, exact_val, fig=None):
    lines = [
        ("Fonction", f"f(x) = {func_str}"),
        ("Intervalle", f"[{a}, {b}]"),
        ("n", str(n)), ("h = (b-a)/n", f"{(b-a)/n:.8f}"),
        ("Résultat numérique", f"{result:.12f}"),
    ]
    if exact_val is not None:
        abs_err = abs(result - exact_val)
        rel_err = abs_err / abs(exact_val) if exact_val != 0 else float("inf")
        lines += [("Valeur exacte", f"{exact_val:.12f}"),
                  ("Erreur absolue", f"{abs_err:.4e}"),
                  ("Erreur relative", f"{rel_err:.4e}")]
    return make_pdf_generic("Rapport Trapèzes", lines, fig)

def make_pdf_cmp(func_str, a, b, n, trap_res, simp_res, exact_val, fig=None):
    lines = [
        ("Fonction", f"f(x) = {func_str}"),
        ("Intervalle", f"[{a}, {b}]"),
        ("n", str(n)),
        ("Résultat Trapèzes", f"{trap_res:.12f}"),
        ("Résultat Simpson", f"{simp_res:.12f}"),
    ]
    if exact_val is not None:
        te = abs(trap_res - exact_val); se = abs(simp_res - exact_val)
        lines += [
            ("Valeur exacte", f"{exact_val:.12f}"),
            ("Erreur Trapèzes", f"{te:.4e}"),
            ("Erreur Simpson", f"{se:.4e}"),
            ("Rapport Trap/Simp", f"{te/se:.2f}×" if se > 0 else "N/A"),
            ("Méthode gagnante", "Simpson 1/3" if se < te else "Trapèzes"),
        ]
    return make_pdf_generic("Rapport Comparaison Trap vs Simpson", lines, fig)

def make_pdf_gj(n_size, A, b_vec, solution, steps, det, rank, cond, rref, fig_sol=None):
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    pdf = FPDF(); pdf.add_page(); pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_fill_color(7, 13, 26); pdf.set_text_color(6, 182, 212)
    pdf.rect(0, 0, 210, 28, 'F'); pdf.set_xy(10, 8)
    pdf.cell(0, 12, "Rapport Gauss-Jordan", ln=True)
    pdf.set_text_color(30, 30, 30); pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 32); pdf.cell(0, 6, f"Date : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4); pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(10, 22, 40); pdf.set_text_color(6, 182, 212)
    pdf.cell(0, 8, f"  Systeme lineaire {n_size}x{n_size}", ln=True, fill=True)
    pdf.set_font("Courier", "", 9); pdf.set_text_color(30, 30, 30); pdf.ln(2)
    col_w = min(30, 150 // n_size)
    for i in range(n_size):
        pdf.set_x(15)
        for j in range(n_size): pdf.cell(col_w, 6, f"{smart_fmt(A[i,j]):>8}", border=1)
        pdf.cell(10, 6, f" | {smart_fmt(b_vec[i])}", ln=True)
    pdf.ln(4); pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(10, 22, 40); pdf.set_text_color(6, 182, 212)
    pdf.cell(0, 8, "  Indicateurs", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 10); pdf.set_text_color(30, 30, 30); pdf.ln(2)
    for label, val in [("Rang", f"{rank}/{n_size}"), ("Determinant", smart_fmt(det) if det is not None else "N/A"),
                        ("Conditionnement", f"{cond:.4e}" if cond is not None else "N/A"),
                        ("Etapes", str(len(steps)))]:
        pdf.cell(70, 7, f"  {label}", border="B"); pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, val, border="B", ln=True); pdf.set_font("Helvetica", "", 10)
    if solution is not None:
        pdf.ln(4); pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(5, 26, 18); pdf.set_text_color(16, 185, 129)
        pdf.cell(0, 8, "  Solution du systeme", ln=True, fill=True)
        pdf.set_font("Courier", "B", 11); pdf.set_text_color(30, 30, 30); pdf.ln(3)
        for i, xi in enumerate(solution): pdf.set_x(20); pdf.cell(0, 7, f"x{i+1}  =  {smart_fmt(xi)}", ln=True)
        if fig_sol is not None:
            img_buf = BytesIO(); fig_sol.savefig(img_buf, format="png", dpi=110, bbox_inches="tight")
            img_buf.seek(0); pdf.add_page()
            pdf.set_font("Helvetica", "B", 12); pdf.set_fill_color(10, 22, 40)
            pdf.set_text_color(6, 182, 212); pdf.cell(0, 8, "  Graphique", ln=True, fill=True)
            pdf.image(img_buf, x=10, w=190)
    return bytes(pdf.output())


# ─────────────────────────────────────────────
#  SHARED SIDEBAR WIDGET: function input
# ─────────────────────────────────────────────
PRESETS = [
    "x**2", "x**3 - 3*x + 1", "1/x", "sqrt(x)", "abs(x)",
    "sin(x)", "cos(x)", "tan(x)", "arctan(x)", "arcsin(x)", "arccos(x)",
    "exp(x)", "exp(-x**2)", "log(x)", "sh(x)", "ch(x)", "th(x)",
    "argsh(x)", "argch(x)", "argth(x)", "1/(1+x**2)"
]

def render_function_sidebar(key_prefix=""):
    """Render function input widgets in sidebar. Returns func_str."""
    func_option = st.radio(T["function_choice"], [T["predefined"], T["custom"]],
                           key=f"{key_prefix}_func_option")
    if func_option == T["predefined"]:
        func_str = st.selectbox(T["select_function"], PRESETS, key=f"{key_prefix}_preset")
    else:
        ck = f"{key_prefix}_custom_func"
        if ck not in st.session_state: st.session_state[ck] = "x**2 * sin(x)"
        def add_sym(sym): st.session_state[ck] += sym
        def clear_func(): st.session_state[ck] = ""
        def del_last(): st.session_state[ck] = st.session_state[ck][:-1]
        func_str = st.text_input(T["enter_function"], key=ck)
        st.markdown("<small>🖩 **Clavier rapide :**</small>", unsafe_allow_html=True)
        pad_4 = [
            [('7','7'),('8','8'),('9','9'),('/','/')],
            [('4','4'),('5','5'),('6','6'),('*','*')],
            [('1','1'),('2','2'),('3','3'),('-','-')],
            [('0','0'),('.','.'),('^','**'),('+','+')],
            [('(','('),(')',')'),('x','x'),('pi','pi')]
        ]
        for r_idx, row in enumerate(pad_4):
            cols = st.columns(4)
            for i, (label, val) in enumerate(row):
                cols[i].button(label, on_click=add_sym, args=(val,), use_container_width=True,
                               key=f"{key_prefix}_num_{r_idx}_{i}_{label}")
        pad_3 = [
            [('sin','sin('),('cos','cos('),('tan','tan(')],
            [('exp','exp('),('log','log('),('sqrt','sqrt(')],
            [('asin','arcsin('),('acos','arccos('),('atan','arctan(')]
        ]
        for r_idx, row in enumerate(pad_3):
            cols = st.columns(3)
            for i, (label, val) in enumerate(row):
                cols[i].button(label, on_click=add_sym, args=(val,), use_container_width=True,
                               key=f"{key_prefix}_func_{r_idx}_{i}_{label}")
        c1, c2 = st.columns(2)
        c1.button("⌫", on_click=del_last, use_container_width=True, key=f"{key_prefix}_del")
        c2.button("C", on_click=clear_func, use_container_width=True, key=f"{key_prefix}_clear")
        st.caption(T["function_hint"])
    return func_str


def apply_dark_axes(ax, fig):
    fig.patch.set_facecolor("#070d1a"); ax.set_facecolor("#070d1a")
    ax.tick_params(colors="#64748b")
    for spine in ax.spines.values(): spine.set_edgecolor("#1e3a5f")
    ax.grid(True, linestyle="--", alpha=0.15, color="#06b6d4")


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: GAUSS-JORDAN
# ═══════════════════════════════════════════════════════════════════════
if st.session_state["page"] == "gj":
    st.subheader(T["gj_title"]); st.caption(T["gj_subtitle"])
    with st.expander(T["gj_about"], expanded=False):
        st.markdown(T["gj_about_text"])
        st.latex(r"""
        \begin{pmatrix}a_{11}&\cdots&a_{1n}\\\vdots&\ddots&\vdots\\a_{n1}&\cdots&a_{nn}\end{pmatrix}
        \begin{pmatrix}b_1\\\vdots\\b_n\end{pmatrix}
        \xrightarrow{\text{Gauss-Jordan}}
        \begin{pmatrix}1&\cdots&0\\\vdots&\ddots&\vdots\\0&\cdots&1\end{pmatrix}
        \begin{pmatrix}x_1\\\vdots\\x_n\end{pmatrix}
        """)
    with st.sidebar:
        st.subheader(T["params"])
        n_size = st.slider(T["gj_size"], min_value=2, max_value=6, value=3)

    st.subheader(T["gj_matrix_a"]); st.caption(T["gj_enter_a"])
    A_input = np.zeros((n_size, n_size))
    defaults_A = {
        2: [[2,1],[5,3]],
        3: [[2,1,-1],[-3,-1,2],[-2,1,2]],
        4: [[2,1,-1,0],[-3,-1,2,1],[-2,1,2,0],[0,1,-1,2]],
        5: [[2,1,-1,0,1],[-3,-1,2,1,0],[-2,1,2,0,-1],[0,1,-1,2,0],[1,0,1,-1,3]],
        6: [[2,1,-1,0,1,0],[-3,-1,2,1,0,1],[-2,1,2,0,-1,0],[0,1,-1,2,0,1],[1,0,1,-1,3,0],[0,1,0,1,-1,2]],
    }
    default_A = defaults_A.get(n_size, np.eye(n_size).tolist())
    default_b = {2:[8,13],3:[8,-11,-3],4:[8,-11,-3,5],5:[8,-11,-3,5,2],6:[8,-11,-3,5,2,1]}
    default_b_val = default_b.get(n_size, [0]*n_size)

    cols_h = st.columns(n_size)
    for j in range(n_size):
        cols_h[j].markdown(
            f'<div style="text-align:center;font-family:IBM Plex Mono;color:#06b6d4;font-size:0.85rem;">x{j+1}</div>',
            unsafe_allow_html=True)
    for i in range(n_size):
        row_cols = st.columns(n_size)
        for j in range(n_size):
            A_input[i, j] = row_cols[j].number_input(
                f"a{i+1}{j+1}", value=float(default_A[i][j]), step=1.0, format="%g",
                label_visibility="collapsed", key=f"A_{i}_{j}")

    st.subheader(T["gj_vector_b"]); st.caption(T["gj_enter_b"])
    b_input = np.zeros(n_size); b_cols = st.columns(n_size)
    for i in range(n_size):
        b_input[i] = b_cols[i].number_input(f"b{i+1}", value=float(default_b_val[i]),
                                             step=1.0, format="%g",
                                             label_visibility="collapsed", key=f"b_{i}")
    with st.expander(T["gj_augmented"], expanded=True):
        Aug_init = np.hstack([A_input, b_input.reshape(-1, 1)])
        st.markdown(fmt_matrix(Aug_init, n_cols=n_size), unsafe_allow_html=True)

    if st.button(T["gj_solve"], type="primary"):
        solution, steps, rref, det, rank, cond, status = gauss_jordan_detailed(A_input, b_input)
        st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(T["gj_rank"], f"{rank} / {n_size}")
        c2.metric(T["gj_det"], smart_fmt(det) if det is not None else "N/A")
        c3.metric(T["gj_cond"], f"{cond:.3e}" if cond is not None else "N/A")
        c4.metric(T["gj_unique"], "✅" if status == "unique" else "❌")
        if cond is not None:
            if cond < 100: st.success(f"✅ Bien conditionné (κ = {cond:.2f})")
            elif cond < 1e6: st.warning(f"⚠️ Modérément conditionné (κ = {cond:.2e})")
            else: st.error(f"❌ Mal conditionné (κ = {cond:.2e})")
        with st.expander(T["gj_what_is_cond"], expanded=False):
            st.markdown(T["gj_cond_explain"])
        if status == "no_solution": st.error(T["gj_no_solution"]); st.stop()
        elif status == "infinite": st.warning(T["gj_inf_solutions"])

        fig_sol_obj = None; residual = None
        if solution is not None:
            st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
            sol_html = f'<div class="solution-card"><h3>{T["gj_solution_direct"]}</h3><div style="display:flex;flex-wrap:wrap;gap:1rem;">'
            for i, xi in enumerate(solution):
                sol_html += (f'<div style="background:#0a2018;border:1px solid #10b981;border-radius:8px;'
                             f'padding:0.7rem 1.2rem;font-family:IBM Plex Mono;min-width:90px;">'
                             f'<div style="color:#64748b;font-size:0.7rem;text-transform:uppercase;">x{i+1}</div>'
                             f'<div style="color:#10b981;font-size:1.25rem;font-weight:700;">{smart_fmt(xi)}</div></div>')
            sol_html += '</div></div>'
            st.markdown(sol_html, unsafe_allow_html=True)
            st.latex(r"x = \begin{pmatrix}" + r"\\".join(smart_fmt(xi) for xi in solution) + r"\end{pmatrix}")

            st.subheader(T["gj_verification"])
            Ax = A_input @ solution; residual = np.linalg.norm(Ax - b_input)
            ver_col1, ver_col2 = st.columns([3, 1])
            with ver_col1:
                ver_data = {"équation": [f"Équation {i+1}" for i in range(n_size)],
                            T["gj_ax"]: [smart_fmt(Ax[i]) for i in range(n_size)],
                            T["gj_bi"]: [smart_fmt(b_input[i]) for i in range(n_size)],
                            T["gj_diff"]: [f"{abs(Ax[i]-b_input[i]):.2e}" for i in range(n_size)]}
                st.dataframe(pd.DataFrame(ver_data), use_container_width=True)
            with ver_col2:
                st.metric(T["gj_residual"], f"{residual:.4e}")
                if residual < 1e-10: st.success(T["gj_residual_negligible"])
                elif residual < 1e-6: st.info(T["gj_residual_low"])
                else: st.warning(T["gj_residual_high"])
            with st.expander(T["gj_what_is_residual"], expanded=False):
                st.markdown(T["gj_residual_explain"])

            fig_sol, ax_sol = plt.subplots(figsize=(max(6, n_size*1.2), 4))
            apply_dark_axes(ax_sol, fig_sol)
            colors = ["#06b6d4" if v >= 0 else "#f59e0b" for v in solution]
            bars = ax_sol.bar([f"x{i+1}" for i in range(n_size)], solution,
                              color=colors, edgecolor="#1e3a5f", linewidth=1.2)
            y_range = max(abs(solution)) if max(abs(solution)) > 0 else 1
            for bar, val in zip(bars, solution):
                ax_sol.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_range * 0.03,
                            smart_fmt(val), ha='center', va='bottom', color="#e2e8f0",
                            fontfamily="monospace", fontsize=9)
            ax_sol.axhline(0, color="#334155", linewidth=0.8)
            ax_sol.set_xlabel("Variables", color="#64748b"); ax_sol.set_ylabel("Valeur", color="#64748b")
            ax_sol.set_title("Solution x — Ax = b", color="#94a3b8", fontfamily="monospace", fontsize=11)
            st.pyplot(fig_sol); fig_sol_obj = fig_sol

        st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
        st.subheader(T["gj_steps"])
        step_icons = {"swap": "🔄", "normalize": "➗", "eliminate": "➖"}
        step_colors = {"swap": "#f59e0b", "normalize": "#06b6d4", "eliminate": "#10b981"}
        for k, step in enumerate(steps):
            stype = step["type"]; icon = step_icons.get(stype, "•"); color = step_colors.get(stype, "#94a3b8")
            op_label = {"swap": T["gj_swap"], "normalize": T["gj_normalize"], "eliminate": T["gj_eliminate"]}.get(stype, stype)
            with st.expander(f"{icon} **{T['gj_step']} {k+1}** — {op_label} — `{step['desc']}`", expanded=(k < 2)):
                col_desc, col_mat = st.columns([1, 2])
                with col_desc:
                    st.markdown(f'<div style="padding:0.8rem;background:#070d1a;border-radius:8px;border-left:3px solid {color};"><span style="color:{color};font-family:IBM Plex Mono;font-size:0.88rem;font-weight:600;">{op_label}</span><br><br><code style="color:#e2e8f0;">{step["desc"]}</code></div>', unsafe_allow_html=True)
                with col_mat:
                    st.markdown(f"**{T['gj_matrix_after']}**")
                    st.markdown(fmt_matrix(step["matrix"], highlight_col=step.get("col"), n_cols=n_size), unsafe_allow_html=True)

        st.subheader(T["gj_rref"])
        st.markdown(fmt_matrix(rref, n_cols=n_size), unsafe_allow_html=True)
        with st.expander(T["gj_what_is_rref"], expanded=False):
            st.markdown(T["gj_rref_explain"])

        st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
        st.subheader(T["gj_summary"])
        synth = {"Indicateur": [T["gj_rank"], T["gj_det"], T["gj_cond"], "Nombre d'étapes", T["gj_residual"]],
                 "Valeur": [f"{rank}/{n_size}", smart_fmt(det) if det else "N/A",
                            f"{cond:.4e}" if cond else "N/A", str(len(steps)),
                            f"{residual:.4e}" if residual is not None else "N/A"]}
        st.dataframe(pd.DataFrame(synth), use_container_width=True)

        st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
        st.subheader(T["download_title"])
        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button(T["download_txt"], data=make_txt_gj(n_size, A_input, b_input, solution, steps, det, rank, cond),
                            file_name="gauss_jordan_report.txt", mime="text/plain")
        dl2.download_button(T["download_csv"], data=make_csv_gj(solution, A_input, b_input),
                            file_name="gauss_jordan_solution.csv", mime="text/csv")
        pdf_gj = make_pdf_gj(n_size, A_input, b_input, solution, steps, det, rank, cond, rref, fig_sol_obj)
        if pdf_gj: dl3.download_button(T["download_pdf"], data=pdf_gj, file_name="gauss_jordan_report.pdf", mime="application/pdf")
        else: dl3.info("PDF: `pip install fpdf2`")


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: SIMPSON 1/3
# ═══════════════════════════════════════════════════════════════════════
elif st.session_state["page"] == "simp":
    st.markdown(f"""
<div class="info-banner">
    <div style="font-family:IBM Plex Mono;font-size:0.78rem;color:#06b6d4;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.5rem;">{T["simp_intro_title"]}</div>
    <div style="font-size:1.0rem;color:#cbd5e1;">{T["simp_intro_text"]}</div>
</div>""", unsafe_allow_html=True)
    st.latex(r"\int_a^b f(x)\,dx")
    st.markdown(f'<div style="color:#94a3b8;font-size:0.92rem;margin-top:0.3rem;">{T["simp_intro_sub"]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.subheader(T["params"])
        func_str = render_function_sidebar("simp")
        col_a, col_b = st.columns(2)
        a = col_a.number_input(T["lower"], value=0.0, step=0.5, format="%g", key="simp_a")
        b = col_b.number_input(T["upper"], value=2.0, step=0.5, format="%g", key="simp_b")
        n_input = st.number_input(T["intervals"], min_value=2, value=6, step=2, key="simp_n")
        n = int(n_input); n_adjusted = False
        if n % 2 != 0: n += 1; n_adjusted = True
        st.divider()
        show_plot  = st.checkbox(T["show_plot"], value=True, key="simp_plt")
        show_table = st.checkbox(T["show_table"], value=False, key="simp_tbl")
        compare_exact = st.checkbox(T["compare_exact"], value=True, key="simp_exact")
        show_conv  = st.checkbox(T["convergence"], value=False, key="simp_conv")
        st.divider()
        run = st.button(T["calculate"], type="primary", use_container_width=True, key="simp_run")

    with st.expander(T["formula_title"], expanded=False):
        st.latex(r"\int_a^b f(x)\,dx \approx \frac{h}{3}\left[f(x_0)+4\sum_{i\text{ impair}}f(x_i)+2\sum_{i\text{ pair}}f(x_i)+f(x_n)\right]")
        st.markdown("avec $h = (b-a)/n$, $n$ pair")

    if run:
        if n_adjusted: st.warning(T["n_must_even"])
        try: evaluate(func_str, np.linspace(a, b, 5))
        except Exception as e: st.error(f"{T['error_func']} : {e}"); st.stop()
        exact_val = exact_integral(func_str, a, b) if compare_exact else None
        if compare_exact and exact_val is None: st.warning(T["error_exact"])
        try: x_vals, y_vals, result, n = simpson_13(func_str, a, b, n)
        except Exception as e: st.error(f"{T['error_calc']} : {e}"); st.stop()
        h_val = (b - a) / n
        st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
        st.subheader(T["simp_params_display"])
        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("a", smart_fmt(a)); pm2.metric("b", smart_fmt(b))
        pm3.metric("n", str(n)); pm4.metric(T["simp_h_label"], f"{h_val:.6f}")
        st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
        st.subheader(T["results"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(T["numerical_result"], f"{result:.10f}")
        if exact_val is not None:
            abs_err = abs(result - exact_val)
            rel_err = abs_err / abs(exact_val) if exact_val != 0 else float("inf")
            c2.metric(T["exact_value"], f"{exact_val:.10f}")
            c3.metric(T["abs_error"], f"{abs_err:.3e}")
            c4.metric(T["rel_error"], f"{rel_err:.3e}")
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            with st.expander(T["btn_abs_error"], expanded=False): st.markdown(T["explain_abs_error"])
        with exp_col2:
            with st.expander(T["btn_rel_error"], expanded=False): st.markdown(T["explain_rel_error"])
        with exp_col3:
            with st.expander(T["btn_convergence"], expanded=False): st.markdown(T["explain_convergence"])
        st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
        fig = None
        if show_plot:
            fig_plot, ax = plt.subplots(figsize=(11, 4.5))
            apply_dark_axes(ax, fig_plot)
            x_fine = np.linspace(a, b, 600); y_fine = evaluate(func_str, x_fine)
            ax.plot(x_fine, y_fine, color="#06b6d4", linewidth=2.2, label=f"f(x) = {func_str}", zorder=4)
            ax.fill_between(x_fine, y_fine, alpha=0.1, color="#06b6d4")
            for i in range(0, len(x_vals)-2, 2):
                coeffs = np.polyfit([x_vals[i], x_vals[i+1], x_vals[i+2]],
                                    [y_vals[i], y_vals[i+1], y_vals[i+2]], 2)
                xp = np.linspace(x_vals[i], x_vals[i+2], 50); yp = np.polyval(coeffs, xp)
                ax.fill_between(xp, yp, alpha=0.35, color="#a78bfa", zorder=2)
                ax.plot(xp, yp, color="#a78bfa", linewidth=1, alpha=0.8, zorder=3)
            ax.plot(x_vals, y_vals, 's', color="#a78bfa", markersize=5, label="Simpson 1/3", zorder=5)
            ax.axhline(0, color="#334155", linewidth=0.8)
            ax.set_xlabel("x", color="#64748b"); ax.set_ylabel("f(x)", color="#64748b")
            ax.set_title(f"Simpson 1/3 — ∫ f(x)dx ≈ {result:.8f}", color="#94a3b8", fontfamily="monospace", fontsize=11)
            ax.legend(facecolor="#070d1a", labelcolor="#e2e8f0", fontsize=9, edgecolor="#1e3a5f")
            st.pyplot(fig_plot); fig = fig_plot
        if show_conv and exact_val is not None:
            ns_c, errs_c = convergence_data(func_str, a, b, exact_val)
            fig_c, ax_c = plt.subplots(figsize=(9, 3.5))
            apply_dark_axes(ax_c, fig_c)
            valid = [(n2, e) for n2, e in zip(ns_c, errs_c) if e > 0 and not np.isnan(e)]
            if valid:
                ns_v, es_v = zip(*valid)
                ax_c.loglog(ns_v, es_v, 'o-', color="#a78bfa", linewidth=2, markersize=4, label="Simpson O(h⁴)")
            ax_c.set_xlabel(T["conv_n"], color="#64748b"); ax_c.set_ylabel(T["conv_err"], color="#64748b")
            ax_c.set_title(T["conv_title"], color="#94a3b8", fontfamily="monospace")
            ax_c.legend(facecolor="#070d1a", labelcolor="#e2e8f0", edgecolor="#1e3a5f")
            st.pyplot(fig_c)
        if show_table:
            st.subheader(T["eval_points"])
            df = pd.DataFrame({"i": range(len(x_vals)), "x_i": x_vals, "f(x_i)": y_vals})
            st.dataframe(df.style.format({"x_i": "{:.8f}", "f(x_i)": "{:.8f}"}), use_container_width=True)
        st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
        st.subheader(T["download_title"])
        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button(T["download_txt"], data=make_txt_simp(func_str, a, b, n, result, exact_val, x_vals, y_vals),
                            file_name="simpson_report.txt", mime="text/plain")
        dl2.download_button(T["download_csv"], data=make_csv_simp(x_vals, y_vals),
                            file_name="simpson_points.csv", mime="text/csv")
        pdf_b = make_pdf_simp(func_str, a, b, n, result, exact_val, fig)
        if pdf_b: dl3.download_button(T["download_pdf"], data=pdf_b, file_name="simpson_report.pdf", mime="application/pdf")
        else: dl3.info("PDF: `pip install fpdf2`")

    st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
    with st.expander(T["about_title"], expanded=False):
        st.markdown(T["about_simp"])


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: TRAPÈZES GÉNÉRALISÉE
# ═══════════════════════════════════════════════════════════════════════
elif st.session_state["page"] == "trap":

    st.markdown(f"""
<div class="info-banner-orange">
    <div style="font-family:IBM Plex Mono;font-size:0.78rem;color:#f59e0b;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.5rem;">{T["trap_intro_title"]}</div>
    <div style="font-size:1.0rem;color:#cbd5e1;">{T["trap_intro_text"]}</div>
</div>""", unsafe_allow_html=True)
    st.latex(r"\int_a^b f(x)\,dx")
    st.markdown(f'<div style="color:#94a3b8;font-size:0.92rem;margin-top:0.3rem;">{T["trap_intro_sub"]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.subheader(T["params"])
        func_str = render_function_sidebar("trap")
        col_a, col_b = st.columns(2)
        a = col_a.number_input(T["lower"], value=0.0, step=0.5, format="%g", key="trap_a")
        b = col_b.number_input(T["upper"], value=2.0, step=0.5, format="%g", key="trap_b")
        n = int(st.number_input(T["intervals"], min_value=1, value=6, step=1, key="trap_n"))
        st.divider()
        show_plot     = st.checkbox(T["show_plot"], value=True, key="trap_plt")
        show_table    = st.checkbox(T["show_table"], value=False, key="trap_tbl")
        compare_exact = st.checkbox(T["compare_exact"], value=True, key="trap_exact")
        show_conv     = st.checkbox(T["convergence"], value=False, key="trap_conv")
        st.divider()
        run = st.button(T["calculate"], type="primary", use_container_width=True, key="trap_run")

    with st.expander(T["trap_formula_title"], expanded=False):
        st.latex(r"\int_a^b f(x)\,dx \approx \frac{h}{2}\left[f(x_0) + 2\sum_{i=1}^{n-1}f(x_i) + f(x_n)\right]")
        st.markdown("avec $h = (b-a)/n$, n quelconque")

    if run:
        try: evaluate(func_str, np.linspace(a, b, 5))
        except Exception as e: st.error(f"{T['error_func']} : {e}"); st.stop()
        exact_val = exact_integral(func_str, a, b) if compare_exact else None
        if compare_exact and exact_val is None: st.warning(T["error_exact"])
        try: x_vals, y_vals, result = trapezoid_rule(func_str, a, b, n)
        except Exception as e: st.error(f"{T['error_calc']} : {e}"); st.stop()
        h_val = (b - a) / n

        st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
        st.subheader(T["trap_params_display"])
        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("a", smart_fmt(a)); pm2.metric("b", smart_fmt(b))
        pm3.metric("n", str(n)); pm4.metric(T["trap_h_label"], f"{h_val:.6f}")

        st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
        st.subheader(T["results"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(T["numerical_result"], f"{result:.10f}")
        if exact_val is not None:
            abs_err = abs(result - exact_val)
            rel_err = abs_err / abs(exact_val) if exact_val != 0 else float("inf")
            c2.metric(T["exact_value"], f"{exact_val:.10f}")
            c3.metric(T["abs_error"], f"{abs_err:.3e}")
            c4.metric(T["rel_error"], f"{rel_err:.3e}")

        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            with st.expander(T["btn_trap_error"], expanded=False): st.markdown(T["explain_trap_error"])
        with exp_col2:
            with st.expander(T["btn_rel_error"], expanded=False): st.markdown(T["explain_rel_error"])
        with exp_col3:
            with st.expander(T["btn_convergence"], expanded=False): st.markdown(T["explain_convergence"])

        st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
        fig = None
        if show_plot:
            fig_plot, ax = plt.subplots(figsize=(11, 4.5))
            apply_dark_axes(ax, fig_plot)
            x_fine = np.linspace(a, b, 600); y_fine = evaluate(func_str, x_fine)
            ax.plot(x_fine, y_fine, color="#06b6d4", linewidth=2.2, label=f"f(x) = {func_str}", zorder=4)
            ax.fill_between(x_fine, y_fine, alpha=0.08, color="#06b6d4")
            for i in range(len(x_vals) - 1):
                ax.fill_between([x_vals[i], x_vals[i+1]], [y_vals[i], y_vals[i+1]],
                                alpha=0.4, color="#f59e0b", zorder=2)
                ax.plot([x_vals[i], x_vals[i]], [0, y_vals[i]], color="#f59e0b",
                        linewidth=0.7, linestyle="--", zorder=3, alpha=0.6)
            ax.plot(x_vals, y_vals, 'o', color="#f59e0b", markersize=6, label="Trapèzes", zorder=5)
            ax.axhline(0, color="#334155", linewidth=0.8)
            ax.set_xlabel("x", color="#64748b"); ax.set_ylabel("f(x)", color="#64748b")
            ax.set_title(f"Trapèzes — ∫ f(x)dx ≈ {result:.8f}", color="#94a3b8", fontfamily="monospace", fontsize=11)
            ax.legend(facecolor="#070d1a", labelcolor="#e2e8f0", fontsize=9, edgecolor="#1e3a5f")
            st.pyplot(fig_plot); fig = fig_plot

        if show_conv and exact_val is not None:
            ns_c, errs_c = convergence_data_trap(func_str, a, b, exact_val)
            fig_c, ax_c = plt.subplots(figsize=(9, 3.5))
            apply_dark_axes(ax_c, fig_c)
            valid = [(n2, e) for n2, e in zip(ns_c, errs_c) if e > 0 and not np.isnan(e)]
            if valid:
                ns_v, es_v = zip(*valid)
                ax_c.loglog(ns_v, es_v, 'o-', color="#f59e0b", linewidth=2, markersize=4, label="Trapèzes O(h²)")
                # Theoretical O(h²) reference line
                ns_arr = np.array(ns_v); ref = es_v[0] * (ns_arr[0] / ns_arr) ** 2
                ax_c.loglog(ns_arr, ref, '--', color="#f59e0b", alpha=0.4, linewidth=1, label="Référence O(h²)")
            ax_c.set_xlabel(T["conv_n"], color="#64748b"); ax_c.set_ylabel(T["conv_err"], color="#64748b")
            ax_c.set_title(T["conv_title"], color="#94a3b8", fontfamily="monospace")
            ax_c.legend(facecolor="#070d1a", labelcolor="#e2e8f0", edgecolor="#1e3a5f")
            st.pyplot(fig_c)

        if show_table:
            st.subheader(T["eval_points"])
            df = pd.DataFrame({"i": range(len(x_vals)), "x_i": x_vals, "f(x_i)": y_vals})
            st.dataframe(df.style.format({"x_i": "{:.8f}", "f(x_i)": "{:.8f}"}), use_container_width=True)

        st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
        st.subheader(T["download_title"])
        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button(T["download_txt"], data=make_txt_trap(func_str, a, b, n, result, exact_val, x_vals, y_vals),
                            file_name="trapeze_report.txt", mime="text/plain")
        dl2.download_button(T["download_csv"], data=make_csv_trap(x_vals, y_vals),
                            file_name="trapeze_points.csv", mime="text/csv")
        pdf_b = make_pdf_trap(func_str, a, b, n, result, exact_val, fig)
        if pdf_b: dl3.download_button(T["download_pdf"], data=pdf_b, file_name="trapeze_report.pdf", mime="application/pdf")
        else: dl3.info("PDF: `pip install fpdf2`")

    st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
    with st.expander(T["trap_about"], expanded=False):
        st.markdown(T["trap_about_text"])
        st.latex(r"|E_T| \leq \frac{(b-a)^3}{12n^2} \max_{[a,b]} |f''(x)|")


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: COMPARAISON TRAPÈZES vs SIMPSON 1/3
# ═══════════════════════════════════════════════════════════════════════
elif st.session_state["page"] == "comp":

    st.markdown(f"""
<div class="info-banner-purple">
    <div style="font-family:IBM Plex Mono;font-size:0.78rem;color:#a78bfa;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.5rem;">📌 Analyse comparative</div>
    <div style="font-size:1.15rem;color:#cbd5e1;font-weight:600;">{T["cmp_title"]}</div>
    <div style="font-size:0.92rem;color:#94a3b8;margin-top:0.4rem;">{T["cmp_subtitle"]}</div>
</div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.subheader(T["params"])
        func_str = render_function_sidebar("cmp")
        col_a, col_b = st.columns(2)
        a = col_a.number_input(T["lower"], value=0.0, step=0.5, format="%g", key="cmp_a")
        b = col_b.number_input(T["upper"], value=2.0, step=0.5, format="%g", key="cmp_b")
        n_raw = int(st.number_input(T["intervals"], min_value=2, value=6, step=2, key="cmp_n",
                                    help="n sera rendu pair pour Simpson 1/3"))
        n = n_raw if n_raw % 2 == 0 else n_raw + 1
        compare_exact = st.checkbox(T["compare_exact"], value=True, key="cmp_exact")
        show_side     = st.checkbox(T["cmp_graph_side"], value=True, key="cmp_side")
        show_conv     = st.checkbox(T["convergence"], value=True, key="cmp_conv")
        st.divider()
        run = st.button(T["cmp_calculate"], type="primary", use_container_width=True, key="cmp_run")

    # ── Theoretical comparison (always visible) ────────────────────
    with st.expander(T["cmp_theory"], expanded=False):
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("### 🟠 Trapèzes")
            st.latex(r"|E_T| \leq \frac{(b-a)^3}{12\,n^2}\,\max|f''(x)|")
            st.markdown("- **Ordre :** O(h²)\n- **Si n×2 :** erreur ÷4\n- **Approximation :** linéaire")
        with col_t2:
            st.markdown("### 🟣 Simpson 1/3")
            st.latex(r"|E_S| \leq \frac{(b-a)^5}{180\,n^4}\,\max|f^{(4)}(x)|")
            st.markdown("- **Ordre :** O(h⁴)\n- **Si n×2 :** erreur ÷16\n- **Approximation :** parabolique")

    if run:
        if n != n_raw: st.info(f"n arrondi à {n} pour Simpson 1/3.")

        try: evaluate(func_str, np.linspace(a, b, 5))
        except Exception as e: st.error(f"{T['error_func']} : {e}"); st.stop()

        exact_val = exact_integral(func_str, a, b) if compare_exact else None
        if compare_exact and exact_val is None: st.warning(T["error_exact"])

        try:
            x_trap, y_trap, trap_res = trapezoid_rule(func_str, a, b, n)
            x_simp, y_simp, simp_res, n = simpson_13(func_str, a, b, n)
        except Exception as e: st.error(f"{T['error_calc']} : {e}"); st.stop()

        h_val = (b - a) / n

        # ── Parameters display ────────────────────────────────────
        st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
        st.subheader(T["cmp_params_display"])
        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("a", smart_fmt(a)); pm2.metric("b", smart_fmt(b))
        pm3.metric("n", str(n)); pm4.metric("h = (b−a)/n", f"{h_val:.6f}")

        # ── Results metrics ───────────────────────────────────────
        st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
        st.subheader(T["cmp_results"])

        if exact_val is not None:
            trap_abs = abs(trap_res - exact_val)
            simp_abs = abs(simp_res - exact_val)
            trap_rel = trap_abs / abs(exact_val) if exact_val != 0 else float("inf")
            simp_rel = simp_abs / abs(exact_val) if exact_val != 0 else float("inf")

            c1, c2, c3 = st.columns(3)
            c1.metric(T["cmp_exact"], f"{exact_val:.10f}")
            c2.metric(T["cmp_trap_res"], f"{trap_res:.10f}",
                      delta=f"err: {trap_abs:.3e}", delta_color="inverse")
            c3.metric(T["cmp_simp_res"], f"{simp_res:.10f}",
                      delta=f"err: {simp_abs:.3e}", delta_color="inverse")

            c4, c5, c6 = st.columns(3)
            c4.metric(T["cmp_ratio_label"], f"{trap_abs/simp_abs:.1f}×" if simp_abs > 0 else "∞")
            c5.metric(T["cmp_trap_err"], f"{trap_abs:.4e}")
            c6.metric(T["cmp_simp_err"], f"{simp_abs:.4e}")

            # ── WINNER BANNER ──────────────────────────────────────
            st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
            st.subheader(T["cmp_winner"])

            ratio = trap_abs / simp_abs if simp_abs > 1e-300 else float("inf")
            if simp_abs < trap_abs:
                msg = T["cmp_simp_wins"].format(ratio)
                pct_better = (trap_abs - simp_abs) / trap_abs * 100
                st.markdown(f"""
<div class="winner-banner">
    <div style="font-size:2.5rem;margin-bottom:0.5rem;">🏆</div>
    <div style="font-family:IBM Plex Mono;font-size:1.3rem;font-weight:700;color:#10b981;">{msg}</div>
    <div style="color:#94a3b8;margin-top:0.6rem;font-size:0.95rem;">
        Simpson est <b style="color:#10b981;">{pct_better:.1f}%</b> plus précis pour cette fonction avec n={n}
    </div>
    <div style="margin-top:1rem;display:flex;justify-content:center;gap:3rem;">
        <div style="text-align:center;">
            <div style="color:#f59e0b;font-family:IBM Plex Mono;font-size:0.8rem;">TRAPÈZES</div>
            <div style="color:#f59e0b;font-size:1.1rem;font-weight:700;">{trap_abs:.4e}</div>
        </div>
        <div style="color:#64748b;font-size:1.5rem;align-self:center;">vs</div>
        <div style="text-align:center;">
            <div style="color:#a78bfa;font-family:IBM Plex Mono;font-size:0.8rem;">SIMPSON 1/3</div>
            <div style="color:#a78bfa;font-size:1.1rem;font-weight:700;">{simp_abs:.4e}</div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)
            elif trap_abs < simp_abs:
                ratio_t = simp_abs / trap_abs if trap_abs > 0 else float("inf")
                st.warning(T["cmp_trap_wins"].format(ratio_t))
            else:
                st.info(T["cmp_equal"])
        else:
            st.info(T["cmp_no_exact"])
            c1, c2 = st.columns(2)
            c1.metric(T["cmp_trap_res"], f"{trap_res:.12f}")
            c2.metric(T["cmp_simp_res"], f"{simp_res:.12f}")
            diff = abs(trap_res - simp_res)
            st.metric("Δ |Trap − Simp|", f"{diff:.6e}")
            trap_abs = simp_abs = trap_rel = simp_rel = None

        # ── OVERLAY GRAPH ─────────────────────────────────────────
        st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
        st.subheader(T["cmp_graph"])

        fig_ov, ax_ov = plt.subplots(figsize=(12, 5))
        apply_dark_axes(ax_ov, fig_ov)
        x_fine = np.linspace(a, b, 800)
        y_fine = evaluate(func_str, x_fine)
        ax_ov.plot(x_fine, y_fine, color="#06b6d4", linewidth=2.8, label=f"f(x) = {func_str}", zorder=6)
        ax_ov.fill_between(x_fine, y_fine, alpha=0.06, color="#06b6d4")

        # Trapezoid fills (orange)
        for i in range(len(x_trap) - 1):
            ax_ov.fill_between([x_trap[i], x_trap[i+1]], [y_trap[i], y_trap[i+1]],
                               alpha=0.28, color="#f59e0b", zorder=2)
        ax_ov.plot(x_trap, y_trap, 'o-', color="#f59e0b", markersize=5, linewidth=1.2,
                   label=f"Trapèzes ≈ {trap_res:.6f}", zorder=5, alpha=0.9)

        # Simpson parabola fills (purple)
        for i in range(0, len(x_simp) - 2, 2):
            try:
                coeffs = np.polyfit([x_simp[i], x_simp[i+1], x_simp[i+2]],
                                    [y_simp[i], y_simp[i+1], y_simp[i+2]], 2)
                xp = np.linspace(x_simp[i], x_simp[i+2], 60)
                yp = np.polyval(coeffs, xp)
                ax_ov.fill_between(xp, yp, alpha=0.25, color="#a78bfa", zorder=3)
                ax_ov.plot(xp, yp, color="#a78bfa", linewidth=1.2, alpha=0.8, zorder=4)
            except: pass
        ax_ov.plot(x_simp, y_simp, 's', color="#a78bfa", markersize=5,
                   label=f"Simpson 1/3 ≈ {simp_res:.6f}", zorder=5)
        ax_ov.axhline(0, color="#334155", linewidth=0.8)
        ax_ov.set_xlabel("x", color="#64748b"); ax_ov.set_ylabel("f(x)", color="#64748b")
        ax_ov.set_title(f"Comparaison visuelle — f(x) = {func_str}  |  n = {n}  |  [a,b] = [{a}, {b}]",
                        color="#94a3b8", fontfamily="monospace", fontsize=11)
        ax_ov.legend(facecolor="#070d1a", labelcolor="#e2e8f0", fontsize=9, edgecolor="#1e3a5f")
        st.pyplot(fig_ov)
        fig_for_pdf = fig_ov

        # ── SIDE BY SIDE GRAPHS ───────────────────────────────────
        if show_side:
            st.subheader(T["cmp_graph_side"])
            fig_side, (ax_tl, ax_sl) = plt.subplots(1, 2, figsize=(12, 4))
            fig_side.patch.set_facecolor("#070d1a")

            # Left: Trapezoid
            ax_tl.set_facecolor("#070d1a")
            ax_tl.plot(x_fine, y_fine, color="#06b6d4", linewidth=2, zorder=5)
            for i in range(len(x_trap) - 1):
                ax_tl.fill_between([x_trap[i], x_trap[i+1]], [y_trap[i], y_trap[i+1]],
                                   alpha=0.45, color="#f59e0b", zorder=2)
                ax_tl.plot([x_trap[i], x_trap[i+1]], [y_trap[i], y_trap[i+1]],
                           color="#f59e0b", linewidth=1.2, zorder=3)
            ax_tl.plot(x_trap, y_trap, 'o', color="#f59e0b", markersize=5, zorder=4)
            ax_tl.axhline(0, color="#334155", linewidth=0.8)
            ax_tl.set_title(f"🟠 Trapèzes ≈ {trap_res:.8f}", color="#f59e0b", fontfamily="monospace", fontsize=10)
            ax_tl.set_xlabel("x", color="#64748b"); ax_tl.set_ylabel("f(x)", color="#64748b")
            ax_tl.tick_params(colors="#64748b")
            for sp in ax_tl.spines.values(): sp.set_edgecolor("#1e3a5f")
            ax_tl.grid(True, linestyle="--", alpha=0.15, color="#f59e0b")

            # Right: Simpson
            ax_sl.set_facecolor("#070d1a")
            ax_sl.plot(x_fine, y_fine, color="#06b6d4", linewidth=2, zorder=5)
            for i in range(0, len(x_simp) - 2, 2):
                try:
                    coeffs = np.polyfit([x_simp[i], x_simp[i+1], x_simp[i+2]],
                                        [y_simp[i], y_simp[i+1], y_simp[i+2]], 2)
                    xp = np.linspace(x_simp[i], x_simp[i+2], 60)
                    yp = np.polyval(coeffs, xp)
                    ax_sl.fill_between(xp, yp, alpha=0.4, color="#a78bfa", zorder=2)
                    ax_sl.plot(xp, yp, color="#a78bfa", linewidth=1.5, zorder=3)
                except: pass
            ax_sl.plot(x_simp, y_simp, 's', color="#a78bfa", markersize=5, zorder=4)
            ax_sl.axhline(0, color="#334155", linewidth=0.8)
            ax_sl.set_title(f"🟣 Simpson 1/3 ≈ {simp_res:.8f}", color="#a78bfa", fontfamily="monospace", fontsize=10)
            ax_sl.set_xlabel("x", color="#64748b"); ax_sl.set_ylabel("f(x)", color="#64748b")
            ax_sl.tick_params(colors="#64748b")
            for sp in ax_sl.spines.values(): sp.set_edgecolor("#1e3a5f")
            ax_sl.grid(True, linestyle="--", alpha=0.15, color="#a78bfa")

            plt.tight_layout(pad=2.0)
            st.pyplot(fig_side)

        # ── COMPARISON TABLE ──────────────────────────────────────
        st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
        st.subheader(T["cmp_table"])

        table_data = {
            T["cmp_method_col"]: [
                "Résultat numérique",
                "Valeur exacte (SymPy)",
                "Erreur absolue",
                "Erreur relative",
                "Ordre de convergence",
                "Réduction erreur si n×2",
                "Type d'approximation",
                "Contrainte sur n",
                "Formule d'erreur",
                "Gagnant (précision)",
            ],
            T["cmp_trap_col"]: [
                f"{trap_res:.10f}",
                f"{exact_val:.10f}" if exact_val is not None else "N/A",
                f"{trap_abs:.4e}" if trap_abs is not None else "N/A",
                f"{trap_rel:.4e}" if trap_rel is not None else "N/A",
                T["cmp_order_trap"],
                T["cmp_div_trap"],
                T["cmp_type_trap"],
                T["cmp_constraint_trap"],
                "(b-a)³ / (12n²)",
                "✅" if (trap_abs is not None and simp_abs is not None and trap_abs < simp_abs) else ("❌" if trap_abs is not None else "N/A"),
            ],
            T["cmp_simp_col"]: [
                f"{simp_res:.10f}",
                f"{exact_val:.10f}" if exact_val is not None else "N/A",
                f"{simp_abs:.4e}" if simp_abs is not None else "N/A",
                f"{simp_rel:.4e}" if simp_rel is not None else "N/A",
                T["cmp_order_simp"],
                T["cmp_div_simp"],
                T["cmp_type_simp"],
                T["cmp_constraint_simp"],
                "(b-a)⁵ / (180n⁴)",
                "✅" if (simp_abs is not None and trap_abs is not None and simp_abs < trap_abs) else ("❌" if simp_abs is not None else "N/A"),
            ],
        }
        df_cmp = pd.DataFrame(table_data)
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)

        # ── CONVERGENCE COMPARISON ────────────────────────────────
        if show_conv and exact_val is not None:
            st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
            st.subheader(T["cmp_conv"])

            ns_t, errs_t = convergence_data_trap(func_str, a, b, exact_val)
            ns_s, errs_s = convergence_data(func_str, a, b, exact_val)

            fig_conv, ax_conv = plt.subplots(figsize=(11, 4.5))
            apply_dark_axes(ax_conv, fig_conv)
            ax_conv.grid(True, linestyle="--", alpha=0.15, color="#94a3b8")

            # Trapezoid
            valid_t = [(n2, e) for n2, e in zip(ns_t, errs_t) if e > 0 and not np.isnan(e)]
            if valid_t:
                nt_v, et_v = zip(*valid_t)
                ax_conv.loglog(nt_v, et_v, 'o-', color="#f59e0b", linewidth=2.2, markersize=4,
                               label="🟠 Trapèzes — O(h²)", zorder=4)
                # O(h²) ref
                nt_arr = np.array(nt_v)
                ref_t = et_v[0] * (nt_arr[0] / nt_arr) ** 2
                ax_conv.loglog(nt_arr, ref_t, '--', color="#f59e0b", alpha=0.35, linewidth=1.2, label="Réf. O(h²)")

            # Simpson
            valid_s = [(n2, e) for n2, e in zip(ns_s, errs_s) if e > 0 and not np.isnan(e)]
            if valid_s:
                ns_v, es_v = zip(*valid_s)
                ax_conv.loglog(ns_v, es_v, 's-', color="#a78bfa", linewidth=2.2, markersize=4,
                               label="🟣 Simpson 1/3 — O(h⁴)", zorder=5)
                # O(h⁴) ref
                ns_arr = np.array(ns_v)
                ref_s = es_v[0] * (ns_arr[0] / ns_arr) ** 4
                ax_conv.loglog(ns_arr, ref_s, '--', color="#a78bfa", alpha=0.35, linewidth=1.2, label="Réf. O(h⁴)")

            ax_conv.set_xlabel(T["conv_n"], color="#64748b", fontsize=11)
            ax_conv.set_ylabel(T["conv_err"], color="#64748b", fontsize=11)
            ax_conv.set_title("Convergence log-log : Trapèzes (pente ≈ −2) vs Simpson (pente ≈ −4)",
                              color="#94a3b8", fontfamily="monospace", fontsize=11)
            ax_conv.legend(facecolor="#070d1a", labelcolor="#e2e8f0", fontsize=9,
                           edgecolor="#1e3a5f", loc="upper right")
            ax_conv.tick_params(colors="#64748b")
            st.pyplot(fig_conv)
            st.caption("💡 Sur ce graphe log-log, Simpson converge ~2 fois plus vite que les Trapèzes "
                       "(pente −4 vs −2). Pour la même précision, Simpson nécessite ~10× moins de points.")

        # ── N REQUIRED FOR PRECISION ──────────────────────────────
        if exact_val is not None and trap_abs is not None and simp_abs is not None:
            st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
            st.subheader("🎯 Estimation : n nécessaire pour une précision cible")
            eps_target = st.select_slider("Précision cible ε",
                                          options=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10],
                                          value=1e-6,
                                          format_func=lambda v: f"{v:.0e}")
            # Estimate from current data using O(h²) and O(h⁴) scaling
            if trap_abs > 0 and simp_abs > 0:
                n_needed_trap = int(np.ceil(n * (trap_abs / eps_target) ** 0.5)) if trap_abs > eps_target else n
                n_needed_simp = int(np.ceil(n * (simp_abs / eps_target) ** 0.25)) if simp_abs > eps_target else n
                # Round simp to even
                if n_needed_simp % 2 != 0: n_needed_simp += 1
                ne1, ne2 = st.columns(2)
                ne1.metric("🟠 n estimé Trapèzes", str(max(n_needed_trap, 2)),
                           help="Estimé via O(h²) depuis les résultats actuels")
                ne2.metric("🟣 n estimé Simpson", str(max(n_needed_simp, 2)),
                           help="Estimé via O(h⁴) depuis les résultats actuels")
                if n_needed_trap > 0 and n_needed_simp > 0:
                    reduction = n_needed_trap / max(n_needed_simp, 1)
                    st.success(f"✅ Simpson nécessite **{reduction:.1f}× moins** de sous-intervalles que les Trapèzes pour atteindre ε = {eps_target:.0e}")

        # ── DOWNLOADS ─────────────────────────────────────────────
        st.markdown('<div class="divider-accent"></div>', unsafe_allow_html=True)
        st.subheader(T["download_title"])
        dl1, dl2, dl3 = st.columns(3)
        txt_cmp = make_txt_cmp(func_str, a, b, n, trap_res, simp_res, exact_val)
        dl1.download_button(T["download_txt"], data=txt_cmp, file_name="comparaison_report.txt", mime="text/plain")
        csv_cmp = make_csv_cmp(func_str, a, b, n, trap_res, simp_res, exact_val, x_trap, y_trap, x_simp, y_simp)
        dl2.download_button(T["download_csv"], data=csv_cmp, file_name="comparaison_points.csv", mime="text/csv")
        pdf_cmp = make_pdf_cmp(func_str, a, b, n, trap_res, simp_res, exact_val, fig_for_pdf)
        if pdf_cmp: dl3.download_button(T["download_pdf"], data=pdf_cmp, file_name="comparaison_report.pdf", mime="application/pdf")
        else: dl3.info("PDF: `pip install fpdf2`")
