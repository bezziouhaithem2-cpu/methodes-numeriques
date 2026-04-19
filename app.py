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
#  SMART NUMBER FORMATTER  (supprime les zéros inutiles)
# ─────────────────────────────────────────────
def smart_fmt(val, max_dec=8):
    """
    Formate un nombre en supprimant les zéros de queue.
    1.000000  → '1'
    1.500000  → '1.5'
    1.540000  → '1.54'
    0.000000  → '0'
    """
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
    """Format scientifique propre pour les très petits nombres."""
    if val == 0:
        return "0"
    if abs(val) < 1e-4 or abs(val) > 1e6:
        return f"{val:.3e}"
    return smart_fmt(val)

# ─────────────────────────────────────────────
#  TRADUCTIONS / TRANSLATIONS / الترجمات
# ─────────────────────────────────────────────
LANG = {
    "fr": {
        "page_title": "Méthodes Numériques",
        "app_title": "📐 Méthodes Numériques Interactives",
        "app_subtitle": "Gauss-Jordan & Simpson 1/3 — Visualisation et analyse détaillée",
        "params": "⚙️ Paramètres",
        "method": "Méthode d'intégration",
        "methods": ["Gauss-Jordan", "Simpson 1/3"],
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
        "download_txt": "📄 Télécharger en TXT",
        "download_csv": "📊 Télécharger en CSV",
        "download_pdf": "📕 Télécharger en PDF",
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
        # Gauss-Jordan
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
        "gj_rref_explain": """
**Qu'est-ce que la RREF ?**

La **Forme Réduite Échelonnée par Lignes** (RREF) est la forme finale de la matrice après élimination de Gauss-Jordan.

**Propriétés :**
- Chaque ligne non nulle commence par un **1 (pivot)**
- Ce **1** est le seul élément non nul dans sa colonne (tous les autres sont 0)
- Les lignes nulles (si elles existent) sont en bas

**Bénéfice principal :** La solution se lit **directement** dans la dernière colonne, sans aucune substitution arrière.  
Exemple : si la 1ère ligne est `[1 0 0 | 5]`, alors directement **x₁ = 5**.
""",
        "gj_solution": "🎯 Solution du système",
        "gj_x": "x",
        "gj_verification": "🔍 Vérification : A·x = b",
        "gj_residual": "Résidu ‖Ax − b‖",
        "gj_residual_explain": """
**Qu'est-ce que le résidu ?**

Le **résidu** est défini par ‖Ax − b‖ (norme euclidienne de la différence).

- Il mesure **à quel point** la solution calculée satisfait le système original.
- **Résidu négligeable** (< 10⁻¹⁰) : la solution est numériquement exacte — les erreurs sont dues uniquement à l'arrondi machine (≈ 10⁻¹⁶).
- **Résidu faible** (< 10⁻⁶) : bonne solution pour la plupart des applications pratiques.
- **Résidu notable** (≥ 10⁻⁶) : attention — le système est peut-être mal conditionné ou la matrice quasi-singulière.
""",
        "gj_det": "Déterminant de A",
        "gj_rank": "Rang de la matrice",
        "gj_cond": "Numéro de cond. κ",
        "gj_cond_explain": """
**Qu'est-ce que le numéro de conditionnement κ (kappa) ?**

Le **numéro de conditionnement** mesure la **sensibilité de la solution** aux petites perturbations des données (erreurs d'arrondi, bruit de mesure).

- κ = σ_max / σ_min  (rapport des valeurs singulières extrêmes)
- Plus κ est grand, plus le système est **instable numériquement**.

| Valeur de κ | Interprétation |
|-------------|---------------|
| κ < 100 | ✅ Bien conditionné — solution fiable |
| 100 ≤ κ < 10⁶ | ⚠️ Modérément conditionné — prudence |
| κ ≥ 10⁶ | ❌ Mal conditionné — solution potentiellement fausse |

**Règle pratique :** On perd environ log₁₀(κ) chiffres significatifs dans la solution. Si κ = 10⁸ et la précision machine est 10⁻¹⁶, la solution n'aura que ~8 chiffres corrects.
""",
        "gj_unique": "Solution unique",
        "gj_no_solution": "❌ Système incompatible (pas de solution)",
        "gj_inf_solutions": "♾️ Infinité de solutions (système sous-déterminé)",
        "gj_singular": "⚠️ Matrice singulière (det = 0)",
        "gj_about": "📘 À propos de la méthode de Gauss-Jordan",
        "gj_about_text": """
**Gauss-Jordan** est une extension de l'élimination de Gauss qui réduit la matrice augmentée à sa **forme échelonnée réduite (RREF)**.

**Algorithme :**
1. Former la matrice augmentée **[A|b]**
2. Pour chaque colonne pivot :
   - Sélectionner le pivot (max partiel pour stabilité numérique)
   - Diviser la ligne pivot pour obtenir un 1 sur la diagonale
   - Éliminer **toutes** les autres lignes (pas seulement en dessous)
3. Lire directement la solution dans la dernière colonne

**Complexité :** O(n³) opérations

**Avantages :**
- Pas de substitution arrière nécessaire
- Permet de calculer l'inverse d'une matrice
- Détection automatique des systèmes incompatibles ou sous-déterminés

**Indicateurs d'analyse :**
- **Déterminant** : produit des pivots (avant normalisation)
- **Rang** : nombre de lignes non nulles dans la RREF
- **Conditionnement** : mesure la sensibilité de la solution aux perturbations
""",
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
        "page_label": "Page active",
        "gj_residual_negligible": "✅ Résidu négligeable — solution numériquement exacte",
        "gj_residual_low": "ℹ️ Résidu faible — bonne précision",
        "gj_residual_high": "⚠️ Résidu notable — vérifiez le conditionnement",
        "gj_what_is_cond": "ℹ️ C'est quoi le numéro de conditionnement ?",
        "gj_what_is_rref": "ℹ️ C'est quoi la RREF et ses bénéfices ?",
        "gj_what_is_residual": "ℹ️ C'est quoi le résidu négligeable ?",
    },
    "en": {
        "page_title": "Numerical Methods",
        "app_title": "📐 Interactive Numerical Methods",
        "app_subtitle": "Gauss-Jordan & Simpson 1/3 — Visualization and detailed analysis",
        "params": "⚙️ Parameters",
        "method": "Method",
        "methods": ["Gauss-Jordan", "Simpson 1/3"],
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
        "download_txt": "📄 Download as TXT",
        "download_csv": "📊 Download as CSV",
        "download_pdf": "📕 Download as PDF",
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
        "gj_rref_explain": """
**What is RREF?**

The **Reduced Row Echelon Form** (RREF) is the final form of the matrix after Gauss-Jordan elimination.

**Properties:**
- Every non-zero row starts with a **1 (pivot/leading 1)**
- That **1** is the only non-zero element in its column (all others are 0)
- Zero rows (if any) are at the bottom

**Main benefit:** The solution can be read **directly** from the last column — no back substitution needed.  
Example: if the 1st row is `[1 0 0 | 5]`, then directly **x₁ = 5**.
""",
        "gj_solution": "🎯 System solution",
        "gj_x": "x",
        "gj_verification": "🔍 Verification: A·x = b",
        "gj_residual": "Residual ‖Ax − b‖",
        "gj_residual_explain": """
**What is the residual?**

The **residual** is defined as ‖Ax − b‖ (Euclidean norm of the difference).

- It measures **how well** the computed solution satisfies the original system.
- **Negligible residual** (< 10⁻¹⁰): numerically exact solution — errors are due only to machine rounding (≈ 10⁻¹⁶).
- **Low residual** (< 10⁻⁶): good solution for most practical applications.
- **Notable residual** (≥ 10⁻⁶): warning — the system may be ill-conditioned or the matrix nearly singular.
""",
        "gj_det": "Determinant of A",
        "gj_rank": "Matrix rank",
        "gj_cond": "Cond. number κ",
        "gj_cond_explain": """
**What is the condition number κ (kappa)?**

The **condition number** measures the **sensitivity of the solution** to small perturbations in the data (rounding errors, measurement noise).

- κ = σ_max / σ_min  (ratio of extreme singular values)
- The larger κ is, the more **numerically unstable** the system.

| Value of κ | Interpretation |
|------------|---------------|
| κ < 100 | ✅ Well-conditioned — reliable solution |
| 100 ≤ κ < 10⁶ | ⚠️ Moderately conditioned — be careful |
| κ ≥ 10⁶ | ❌ Ill-conditioned — solution potentially unreliable |

**Practical rule:** You lose about log₁₀(κ) significant digits in the solution. If κ = 10⁸ and machine precision is 10⁻¹⁶, the solution has only ~8 correct digits.
""",
        "gj_unique": "Unique solution",
        "gj_no_solution": "❌ Inconsistent system (no solution)",
        "gj_inf_solutions": "♾️ Infinitely many solutions (underdetermined system)",
        "gj_singular": "⚠️ Singular matrix (det = 0)",
        "gj_about": "📘 About the Gauss-Jordan method",
        "gj_about_text": """
**Gauss-Jordan** is an extension of Gaussian elimination that reduces the augmented matrix to its **Reduced Row Echelon Form (RREF)**.

**Algorithm:**
1. Form the augmented matrix **[A|b]**
2. For each pivot column:
   - Select the pivot (partial max for numerical stability)
   - Divide the pivot row to get a 1 on the diagonal
   - Eliminate **all** other rows (not just below)
3. Read the solution directly from the last column

**Complexity:** O(n³) operations

**Advantages:**
- No back substitution needed
- Allows computing the matrix inverse
- Automatic detection of inconsistent or underdetermined systems
""",
        "gj_formula": "General form of the system",
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
        "page_label": "Active page",
        "gj_residual_negligible": "✅ Negligible residual — numerically exact solution",
        "gj_residual_low": "ℹ️ Low residual — good precision",
        "gj_residual_high": "⚠️ Notable residual — check conditioning",
        "gj_what_is_cond": "ℹ️ What is the condition number?",
        "gj_what_is_rref": "ℹ️ What is RREF and its benefits?",
        "gj_what_is_residual": "ℹ️ What is a negligible residual?",
    },
    "ar": {
        "page_title": "الطرق العددية",
        "app_title": "📐 الطرق العددية التفاعلية",
        "app_subtitle": "غاوس-جوردان و سيمبسون 1/3 — تصور وتحليل مفصل",
        "params": "⚙️ الإعدادات",
        "method": "الطريقة",
        "methods": ["غاوس-جوردان", "سيمبسون 1/3"],
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
        "formula_simp": r"$\int_a^b f(x)\,dx \approx \frac{h}{3}\left[f(x_0)+4\sum_{\text{فردي}}f(x_i)+2\sum_{\text{زوجي}}f(x_i)+f(x_n)\right]$",
        "download_title": "⬇️ تحميل النتائج",
        "download_txt": "📄 تحميل كـ TXT",
        "download_csv": "📊 تحميل كـ CSV",
        "download_pdf": "📕 تحميل كـ PDF",
        "error_func": "خطأ في الدالة المدخلة",
        "error_calc": "خطأ في الحساب",
        "error_exact": "تعذّر حساب القيمة الدقيقة.",
        "error_n_even": "يجب أن يكون n زوجياً لـ Simpson 1/3.",
        "conv_title": "📉 التقارب: الخطأ مقابل n",
        "conv_n": "n (الأجزاء الفرعية)",
        "conv_err": "الخطأ المطلق",
        "language": "🌐 اللغة / Langue / Language",
        "n_must_even": "⚠️ يجب أن يكون n زوجياً لسيمبسون. تم تقريب n.",
        "exact_na": "غير متاح (القيمة الدقيقة لم تُحسب)",
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
        "gj_rref_explain": """
**ما هي RREF ؟**

**الصورة المختزلة للدرج** (RREF) هي الشكل النهائي للمصفوفة بعد حذف غاوس-جوردان.

**الخصائص :**
- كل صف غير صفري يبدأ بـ **1 (محور)**
- هذا الـ **1** هو العنصر الوحيد غير الصفري في عمودِه (جميع العناصر الأخرى = 0)
- الصفوف الصفرية (إن وُجدت) في الأسفل

**الفائدة الرئيسية :** الحل يُقرأ **مباشرةً** من العمود الأخير بدون أي إحلال عكسي.  
مثال: إذا كان الصف الأول `[1 0 0 | 5]` فمباشرةً **x₁ = 5**.
""",
        "gj_solution": "🎯 حل النظام",
        "gj_x": "x",
        "gj_verification": "🔍 التحقق: A·x = b",
        "gj_residual": "البقايا ‖Ax − b‖",
        "gj_residual_explain": """
**ما هو البقايا (Residual) ؟**

**البقايا** يُعرَّف بـ ‖Ax − b‖ (المعيار الإقليدي للفرق).

- يقيس **مدى دقة** الحل المحسوب في تحقيق النظام الأصلي.
- **بقايا ضئيلة** (< 10⁻¹⁰) : الحل دقيق عددياً — الأخطاء ناتجة فقط عن تقريب الحاسوب (≈ 10⁻¹⁶).
- **بقايا منخفضة** (< 10⁻⁶) : حل جيد لمعظم التطبيقات العملية.
- **بقايا ملحوظة** (≥ 10⁻⁶) : تحذير — قد يكون النظام سيء التكييف أو المصفوفة شبه شاذة.
""",
        "gj_det": "محدد المصفوفة A",
        "gj_rank": "رتبة المصفوفة",
        "gj_cond": "رقم الشرط κ",
        "gj_cond_explain": """
**ما هو رقم الشرط κ (كابا) ؟**

**رقم الشرط** يقيس **حساسية الحل** للاضطرابات الصغيرة في البيانات (أخطاء التقريب، ضوضاء القياس).

- κ = σ_max / σ_min  (نسبة القيم الشاذة القصوى)
- كلما كان κ أكبر، كان النظام أكثر **عدم استقراراً عددياً**.

| قيمة κ | التفسير |
|--------|---------|
| κ < 100 | ✅ جيد التكييف — حل موثوق |
| 100 ≤ κ < 10⁶ | ⚠️ متوسط التكييف — توخَّ الحذر |
| κ ≥ 10⁶ | ❌ سيء التكييف — الحل قد يكون غير موثوق |

**قاعدة عملية:** تفقد نحو log₁₀(κ) رقماً دقيقاً في الحل.
""",
        "gj_unique": "حل وحيد",
        "gj_no_solution": "❌ نظام غير متوافق (لا يوجد حل)",
        "gj_inf_solutions": "♾️ حلول لا نهائية (نظام ناقص التحديد)",
        "gj_singular": "⚠️ مصفوفة شاذة (det = 0)",
        "gj_about": "📘 حول طريقة غاوس-جوردان",
        "gj_about_text": """
**غاوس-جوردان** امتداد لطريقة الحذف الغاوسي تختزل المصفوفة المعززة إلى **الصورة المختزلة للدرج (RREF)**.

**الخوارزمية:**
1. تشكيل المصفوفة المعززة **[A|b]**
2. لكل عمود محوري:
   - اختيار المحور (أقصى جزئي للاستقرار العددي)
   - قسمة صف المحور للحصول على 1 في القطر
   - حذف **جميع** الصفوف الأخرى
3. قراءة الحل مباشرة من العمود الأخير

**التعقيد:** O(n³) عملية
""",
        "gj_formula": "الصيغة العامة للنظام",
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
        "page_label": "الصفحة النشطة",
        "gj_residual_negligible": "✅ بقايا ضئيلة — الحل دقيق عددياً",
        "gj_residual_low": "ℹ️ بقايا منخفضة — دقة جيدة",
        "gj_residual_high": "⚠️ بقايا ملحوظة — تحقق من التكييف",
        "gj_what_is_cond": "ℹ️ ما هو رقم الشرط؟",
        "gj_what_is_rref": "ℹ️ ما هي RREF وفوائدها؟",
        "gj_what_is_residual": "ℹ️ ما معنى البقايا الضئيلة؟",
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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.03em;
}
.stButton > button {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #e2e8f0;
    border: 1px solid #4a90d9;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 0.6em 1.4em;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: #4a90d9;
    color: #fff;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(74,144,217,0.4);
}
.metric-box {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
    border: 1px solid #4a90d933;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin: 0.5rem 0;
}
.stMetric label { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; }
div[data-testid="stExpander"] { border: 1px solid #4a90d922; border-radius: 8px; }
.block-container { padding-top: 2rem; }
.matrix-step {
    background: #0d1b2a;
    border: 1px solid #4a90d933;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}
.step-badge {
    display: inline-block;
    background: #4a90d9;
    color: white;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 0.5rem;
}
.solution-box {
    background: linear-gradient(135deg, #0a2e1a, #0d1b2a);
    border: 1px solid #2ecc71;
    border-radius: 10px;
    padding: 1.2rem;
    margin: 0.5rem 0;
}
.explain-box {
    background: linear-gradient(135deg, #0d1b2a, #0f2040);
    border: 1px solid #4a90d955;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
}
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
    st.markdown('<style>body, .stMarkdown, .stText { direction: rtl; text-align: right; }</style>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.title(T["app_title"])
st.caption(T["app_subtitle"])
st.divider()

# ─────────────────────────────────────────────
#  NAVIGATION
# ─────────────────────────────────────────────
nav_col1, nav_col2, nav_col3 = st.columns([2, 2, 6])
with nav_col1:
    if st.button(T["nav_gj"], use_container_width=True,
                 type="primary" if st.session_state["page"] == "gj" else "secondary"):
        st.session_state["page"] = "gj"
        st.rerun()
with nav_col2:
    if st.button(T["nav_simp"], use_container_width=True,
                 type="primary" if st.session_state["page"] == "simp" else "secondary"):
        st.session_state["page"] = "simp"
        st.rerun()

st.divider()

# ═══════════════════════════════════════════════════════════════════════
#  MATH HELPERS — GAUSS-JORDAN
# ═══════════════════════════════════════════════════════════════════════

def fmt_matrix(M, highlight_col=None, n_cols=None):
    """Return a nicely formatted augmented matrix as HTML table."""
    rows, cols = M.shape
    sep_col = n_cols

    html = '<table style="border-collapse:collapse;font-family:IBM Plex Mono,monospace;font-size:0.82rem;margin:0.4rem 0;">'
    for i in range(rows):
        html += "<tr>"
        for j in range(cols):
            border_left = "border-left:2px solid #4a90d9;" if (sep_col is not None and j == sep_col) else ""
            bg = "background:#4a90d922;" if (highlight_col is not None and j == highlight_col) else ""
            val = M[i, j]
            # Clean display: remove trailing zeros
            rounded = round(val, 6)
            if abs(rounded) < 1e-9:
                txt = "0"
            elif rounded == int(rounded) and abs(rounded) < 1e9:
                txt = f"{int(rounded):+d}"
            else:
                s = f"{rounded:+.5f}".rstrip('0')
                if s.endswith('.') or s.endswith('+') or s.endswith('-'):
                    s += '0'
                txt = s
            html += f'<td style="padding:3px 10px;{border_left}{bg}color:#e2e8f0;min-width:70px;text-align:right;">{txt}</td>'
        html += "</tr>"
    html += "</table>"
    return html


def gauss_jordan_detailed(A_orig, b_orig):
    n = len(b_orig)
    Aug = np.hstack([A_orig.copy().astype(float), b_orig.copy().reshape(-1, 1).astype(float)])

    steps = []
    det_sign = 1
    pivot_row = 0

    for col in range(n):
        max_row = pivot_row + np.argmax(np.abs(Aug[pivot_row:, col]))
        if abs(Aug[max_row, col]) < 1e-12:
            continue

        if max_row != pivot_row:
            Aug[[pivot_row, max_row]] = Aug[[max_row, pivot_row]]
            det_sign *= -1
            steps.append({
                "type": "swap",
                "col": col,
                "pivot_row": pivot_row,
                "desc": f"R{pivot_row+1} ↔ R{max_row+1}",
                "matrix": Aug.copy(),
            })

        pivot_val = Aug[pivot_row, col]
        Aug[pivot_row] = Aug[pivot_row] / pivot_val
        steps.append({
            "type": "normalize",
            "col": col,
            "pivot_row": pivot_row,
            "desc": f"R{pivot_row+1} ← R{pivot_row+1} / ({smart_fmt(pivot_val)})",
            "matrix": Aug.copy(),
            "pivot_val": pivot_val,
        })

        for row in range(n):
            if row == pivot_row:
                continue
            factor = Aug[row, col]
            if abs(factor) < 1e-12:
                continue
            Aug[row] = Aug[row] - factor * Aug[pivot_row]
            steps.append({
                "type": "eliminate",
                "col": col,
                "target_row": row,
                "pivot_row": pivot_row,
                "desc": f"R{row+1} ← R{row+1} − ({smart_fmt(factor)})·R{pivot_row+1}",
                "matrix": Aug.copy(),
                "factor": factor,
            })

        pivot_row += 1

    rref = Aug.copy()
    rank = sum(1 for i in range(n) if np.any(np.abs(rref[i, :n]) > 1e-10))

    try:
        det = float(np.linalg.det(A_orig.astype(float)))
    except Exception:
        det = None

    try:
        cond = float(np.linalg.cond(A_orig.astype(float)))
    except Exception:
        cond = None

    status = "unique"
    solution = None

    if rank < n:
        for i in range(rank, n):
            if abs(rref[i, n]) > 1e-10:
                status = "no_solution"
                return None, steps, rref, det, rank, cond, status
        status = "infinite"
        return None, steps, rref, det, rank, cond, status

    solution = rref[:, n].copy()
    return solution, steps, rref, det, rank, cond, status


# ═══════════════════════════════════════════════════════════════════════
#  MATH HELPERS — SIMPSON
# ═══════════════════════════════════════════════════════════════════════
_SAFE_NAMES = {
    'pi': np.pi, 'e': np.e,
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
    'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
    'sinh': np.sinh, 'sh': np.sinh,
    'cosh': np.cosh, 'ch': np.cosh,
    'tanh': np.tanh, 'th': np.tanh,
    'asinh': np.arcsinh, 'argsh': np.arcsinh,
    'acosh': np.arccosh, 'argch': np.arccosh,
    'atanh': np.arctanh, 'argth': np.arctanh,
    'exp': np.exp, 'log': np.log, 'ln': np.log,
    'log10': np.log10, 'sqrt': np.sqrt, 'abs': np.abs,
    'floor': np.floor, 'ceil': np.ceil, 'sign': np.sign,
}

def evaluate(func_str, x):
    ns = dict(_SAFE_NAMES)
    ns['x'] = x
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
    except Exception:
        return None

def simpson_13(func_str, a, b, n):
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = evaluate(func_str, x)
    result = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
    return x, y, result, n

def convergence_data(func_str, a, b, exact):
    ns = list(range(2, 52, 2))
    errors = []
    for ni in ns:
        try:
            _, _, res, _ = simpson_13(func_str, a, b, ni)
            errors.append(abs(res - exact))
        except Exception:
            errors.append(np.nan)
    return ns, errors

# ─────────────────────────────────────────────
#  DOWNLOAD HELPERS
# ─────────────────────────────────────────────
def make_txt_simp(func_str, a, b, n, result, exact_val, x_vals, y_vals):
    buf = StringIO()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    buf.write("=" * 52 + "\n")
    buf.write(f"  RAPPORT SIMPSON 1/3 — {now}\n")
    buf.write("=" * 52 + "\n\n")
    buf.write(f"  Fonction        : f(x) = {func_str}\n")
    buf.write(f"  Intervalle      : [{a}, {b}]\n")
    buf.write(f"  Sous-intervalles: n = {n}\n\n")
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
    if x_vals is None:
        return b""
    df = pd.DataFrame({"x_i": x_vals, "f(x_i)": y_vals})
    return df.to_csv(index=True).encode("utf-8")

def make_txt_gj(n_size, A, b_vec, solution, steps, det, rank, cond):
    buf = StringIO()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    buf.write("=" * 60 + "\n")
    buf.write(f"  RAPPORT GAUSS-JORDAN — {now}\n")
    buf.write("=" * 60 + "\n\n")
    buf.write(f"  Taille du système: {n_size}×{n_size}\n\n")
    buf.write("  Matrice A:\n")
    for row in A:
        buf.write("  " + "  ".join(f"{v:>10.5f}" for v in row) + "\n")
    buf.write("\n  Vecteur b:\n")
    buf.write("  " + "  ".join(f"{v:>10.5f}" for v in b_vec) + "\n\n")
    if det is not None:
        buf.write(f"  Déterminant     : {smart_fmt(det)}\n")
    buf.write(f"  Rang            : {rank}\n")
    if cond is not None:
        buf.write(f"  Conditionnement : {cond:.4e}\n")
    buf.write(f"  Nombre d'étapes : {len(steps)}\n\n")
    if solution is not None:
        buf.write("  Solution:\n")
        for i, xi in enumerate(solution):
            buf.write(f"    x{i+1} = {smart_fmt(xi)}\n")
        buf.write("\n  Vérification A·x:\n")
        ax = A @ solution
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
            rows.append({
                "variable": f"x{i+1}",
                "valeur": xi,
                "A·x_i": ax[i],
                "b_i": b_vec[i],
                "résidu": abs(ax[i] - b_vec[i]),
            })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

def make_pdf_gj(n_size, A, b_vec, solution, steps, det, rank, cond, rref, fig_sol=None):
    """Generate a PDF report for Gauss-Jordan results."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Header ──────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_fill_color(15, 52, 96)
    pdf.set_text_color(255, 255, 255)
    pdf.rect(0, 0, 210, 28, 'F')
    pdf.set_xy(10, 8)
    pdf.cell(0, 12, "Rapport Gauss-Jordan", ln=True)
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 32)
    pdf.cell(0, 6, f"Date : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    # ── System parameters ────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(230, 240, 255)
    pdf.set_text_color(15, 52, 96)
    pdf.cell(0, 8, f"  Systeme lineaire {n_size}x{n_size}", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 30, 30)
    pdf.ln(2)

    # Matrix A
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "  Matrice A :", ln=True)
    pdf.set_font("Courier", "", 9)
    col_w = min(30, 150 // n_size)
    for i in range(n_size):
        pdf.set_x(15)
        for j in range(n_size):
            pdf.cell(col_w, 6, f"{smart_fmt(A[i,j]):>8}", border=1)
        pdf.cell(10, 6, f" | {smart_fmt(b_vec[i])}", ln=True)
    pdf.ln(4)

    # ── Analysis indicators ──────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(230, 240, 255)
    pdf.set_text_color(15, 52, 96)
    pdf.cell(0, 8, "  Indicateurs d'analyse", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.ln(2)
    indicators = [
        ("Rang", f"{rank} / {n_size}"),
        ("Determinant", smart_fmt(det) if det is not None else "N/A"),
        ("Numero de conditionnement", f"{cond:.4e}" if cond is not None else "N/A"),
        ("Nombre d'etapes", str(len(steps))),
    ]
    for label, val in indicators:
        pdf.cell(70, 7, f"  {label}", border="B")
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, val, border="B", ln=True)
        pdf.set_font("Helvetica", "", 10)
    pdf.ln(4)

    # ── Solution ─────────────────────────────────────────────────────
    if solution is not None:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(20, 80, 40)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 8, "  Solution du systeme", ln=True, fill=True)
        pdf.set_font("Courier", "B", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.ln(3)
        for i, xi in enumerate(solution):
            pdf.set_x(20)
            pdf.cell(0, 7, f"x{i+1}  =  {smart_fmt(xi)}", ln=True)
        pdf.ln(4)

        # ── Verification table ───────────────────────────────────────
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(230, 240, 255)
        pdf.set_text_color(15, 52, 96)
        pdf.cell(0, 8, "  Verification : A.x = b", ln=True, fill=True)
        pdf.set_text_color(30, 30, 30)
        pdf.ln(2)
        ax = A @ solution
        residual = np.linalg.norm(ax - b_vec)

        # Table header
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(200, 220, 255)
        for header in ["Equation", "A.x_i", "b_i", "|A.x_i - b_i|"]:
            pdf.cell(45, 6, header, border=1, fill=True)
        pdf.ln()

        pdf.set_font("Courier", "", 9)
        for i in range(n_size):
            diff = abs(ax[i] - b_vec[i])
            pdf.cell(45, 6, f"Equation {i+1}", border=1)
            pdf.cell(45, 6, smart_fmt(ax[i]), border=1)
            pdf.cell(45, 6, smart_fmt(b_vec[i]), border=1)
            pdf.cell(45, 6, f"{diff:.2e}", border=1)
            pdf.ln()
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, f"  Residu global ||Ax - b|| = {residual:.4e}", ln=True)
        pdf.ln(4)

        # ── Solution chart ───────────────────────────────────────────
        if fig_sol is not None:
            img_buf = BytesIO()
            fig_sol.savefig(img_buf, format="png", dpi=110, bbox_inches="tight")
            img_buf.seek(0)
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_fill_color(230, 240, 255)
            pdf.set_text_color(15, 52, 96)
            pdf.cell(0, 8, "  Graphique de la solution", ln=True, fill=True)
            pdf.image(img_buf, x=10, w=190)

    return bytes(pdf.output())


def make_pdf_simp(func_str, a, b, n, result, exact_val, fig=None):
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_fill_color(15, 52, 96)
    pdf.set_text_color(255, 255, 255)
    pdf.rect(0, 0, 210, 28, 'F')
    pdf.set_xy(10, 8)
    pdf.cell(0, 12, "Rapport Simpson 1/3", ln=True)
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 32)
    pdf.cell(0, 6, f"Date : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(230, 240, 255)
    pdf.set_text_color(15, 52, 96)
    pdf.cell(0, 8, "  Parametres", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 30, 30)
    for label, val in [("Fonction", f"f(x) = {func_str}"), ("Intervalle", f"[{a}, {b}]"), ("n", str(n))]:
        pdf.cell(55, 7, f"  {label}", border="B")
        pdf.cell(0, 7, val, border="B", ln=True)
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(230, 240, 255)
    pdf.set_text_color(15, 52, 96)
    pdf.cell(0, 8, "  Resultats", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(55, 8, "  Resultat numerique")
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, f"{result:.12f}", ln=True)
    if exact_val is not None:
        abs_err = abs(result - exact_val)
        rel_err = abs_err / abs(exact_val) if exact_val != 0 else float("inf")
        pdf.set_font("Helvetica", "", 10)
        for label, val in [("Valeur exacte", f"{exact_val:.12f}"), ("Erreur absolue", f"{abs_err:.4e}"), ("Erreur relative", f"{rel_err:.4e}")]:
            pdf.cell(55, 7, f"  {label}")
            pdf.cell(0, 7, val, ln=True)
    if fig is not None:
        img_buf = BytesIO()
        fig.savefig(img_buf, format="png", dpi=110, bbox_inches="tight")
        img_buf.seek(0)
        pdf.ln(8)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(230, 240, 255)
        pdf.set_text_color(15, 52, 96)
        pdf.cell(0, 8, "  Graphique", ln=True, fill=True)
        pdf.image(img_buf, x=10, w=190)
    return bytes(pdf.output())


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: GAUSS-JORDAN
# ═══════════════════════════════════════════════════════════════════════
if st.session_state["page"] == "gj":

    st.subheader(T["gj_title"])
    st.caption(T["gj_subtitle"])

    with st.expander(T["gj_about"], expanded=False):
        st.markdown(T["gj_about_text"])
        st.latex(r"""
        \begin{pmatrix}
        a_{11} & a_{12} & \cdots & a_{1n} & \bigg| & b_1 \\
        a_{21} & a_{22} & \cdots & a_{2n} & \bigg| & b_2 \\
        \vdots & \vdots & \ddots & \vdots & \bigg| & \vdots \\
        a_{n1} & a_{n2} & \cdots & a_{nn} & \bigg| & b_n
        \end{pmatrix}
        \xrightarrow{\text{Gauss-Jordan}}
        \begin{pmatrix}
        1 & 0 & \cdots & 0 & \bigg| & x_1 \\
        0 & 1 & \cdots & 0 & \bigg| & x_2 \\
        \vdots & \vdots & \ddots & \vdots & \bigg| & \vdots \\
        0 & 0 & \cdots & 1 & \bigg| & x_n
        \end{pmatrix}
        """)

    # ── Sidebar parameters ───────────────────────────────────────────
    with st.sidebar:
        st.subheader(T["params"])
        n_size = st.slider(T["gj_size"], min_value=2, max_value=6, value=3)

    # ── Matrix A input ───────────────────────────────────────────────
    st.subheader(T["gj_matrix_a"])
    st.caption(T["gj_enter_a"])

    A_input = np.zeros((n_size, n_size))
    defaults_A = {
        2: [[2, 1], [5, 3]],
        3: [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]],
        4: [[2, 1, -1, 0], [-3, -1, 2, 1], [-2, 1, 2, 0], [0, 1, -1, 2]],
        5: [[2,1,-1,0,1],[-3,-1,2,1,0],[-2,1,2,0,-1],[0,1,-1,2,0],[1,0,1,-1,3]],
        6: [[2,1,-1,0,1,0],[-3,-1,2,1,0,1],[-2,1,2,0,-1,0],[0,1,-1,2,0,1],[1,0,1,-1,3,0],[0,1,0,1,-1,2]],
    }
    default_A = defaults_A.get(n_size, np.eye(n_size).tolist())
    default_b = {2:[8,13], 3:[8,-11,-3], 4:[8,-11,-3,5], 5:[8,-11,-3,5,2], 6:[8,-11,-3,5,2,1]}
    default_b_val = default_b.get(n_size, [0]*n_size)

    cols_header = st.columns(n_size)
    for j in range(n_size):
        cols_header[j].markdown(f"<center>**x{j+1}**</center>", unsafe_allow_html=True)

    for i in range(n_size):
        row_cols = st.columns(n_size)
        for j in range(n_size):
            A_input[i, j] = row_cols[j].number_input(
                f"a{i+1}{j+1}", value=float(default_A[i][j]),
                step=1.0, format="%g",
                label_visibility="collapsed",
                key=f"A_{i}_{j}"
            )

    st.subheader(T["gj_vector_b"])
    st.caption(T["gj_enter_b"])
    b_input = np.zeros(n_size)
    b_cols = st.columns(n_size)
    for i in range(n_size):
        b_input[i] = b_cols[i].number_input(
            f"b{i+1}", value=float(default_b_val[i]),
            step=1.0, format="%g",
            label_visibility="collapsed",
            key=f"b_{i}"
        )

    with st.expander(T["gj_augmented"], expanded=True):
        Aug_init = np.hstack([A_input, b_input.reshape(-1, 1)])
        st.markdown(fmt_matrix(Aug_init, n_cols=n_size), unsafe_allow_html=True)

    run_gj = st.button(T["gj_solve"], type="primary", use_container_width=False)

    if run_gj:
        solution, steps, rref, det, rank, cond, status = gauss_jordan_detailed(A_input, b_input)

        st.divider()

        # ── Analysis cards ───────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(T["gj_rank"], f"{rank} / {n_size}")
        c2.metric(T["gj_det"], smart_fmt(det) if det is not None else "N/A")
        c3.metric(T["gj_cond"], f"{cond:.3e}" if cond is not None else "N/A")
        c4.metric(T["gj_unique"], "✅" if status == "unique" else "❌")

        # Condition number status + explanation
        if cond is not None:
            if cond < 100:
                st.success(f"✅ Système bien conditionné (κ = {cond:.2f} < 100)")
            elif cond < 1e6:
                st.warning(f"⚠️ Système modérément conditionné (κ = {cond:.2e})")
            else:
                st.error(f"❌ Système mal conditionné (κ = {cond:.2e} ≥ 10⁶) — solution potentiellement instable")

        with st.expander(T["gj_what_is_cond"], expanded=False):
            st.markdown(T["gj_cond_explain"])

        # ── Status ───────────────────────────────────────────────────
        if status == "no_solution":
            st.error(T["gj_no_solution"])
            st.stop()
        elif status == "infinite":
            st.warning(T["gj_inf_solutions"])

        # ── Detailed steps ───────────────────────────────────────────
        st.subheader(T["gj_steps"])
        step_icons = {"swap": "🔄", "normalize": "➗", "eliminate": "➖"}
        step_colors = {"swap": "#f5a623", "normalize": "#4a90d9", "eliminate": "#2ecc71"}

        for k, step in enumerate(steps):
            stype = step["type"]
            icon = step_icons.get(stype, "•")
            color = step_colors.get(stype, "#aaa")
            op_label = {
                "swap": T["gj_swap"],
                "normalize": T["gj_normalize"],
                "eliminate": T["gj_eliminate"],
            }.get(stype, stype)

            with st.expander(
                f"{icon} **{T['gj_step']} {k+1}** — {op_label} — `{step['desc']}`",
                expanded=(k < 3)
            ):
                col_desc, col_mat = st.columns([1, 2])
                with col_desc:
                    st.markdown(f"""
<div style="padding:0.8rem;background:#0d1b2a;border-radius:8px;border-left:3px solid {color};">
<span style="color:{color};font-family:IBM Plex Mono;font-size:0.9rem;font-weight:600;">{op_label}</span><br><br>
<code style="color:#e2e8f0;">{step['desc']}</code><br><br>
""", unsafe_allow_html=True)
                    if stype == "normalize":
                        st.markdown(f"""<code style="color:#aac;">Pivot = {smart_fmt(step.get('pivot_val', 0))}</code>""", unsafe_allow_html=True)
                    elif stype == "eliminate":
                        st.markdown(f"""<code style="color:#aac;">Facteur = {smart_fmt(step.get('factor', 0))}</code>""", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_mat:
                    st.markdown(f"**{T['gj_matrix_after']}**")
                    st.markdown(fmt_matrix(step["matrix"], highlight_col=step.get("col"), n_cols=n_size), unsafe_allow_html=True)

        # ── RREF ────────────────────────────────────────────────────
        st.subheader(T["gj_rref"])
        st.markdown(fmt_matrix(rref, n_cols=n_size), unsafe_allow_html=True)
        st.latex(r"\text{RREF}(A|b) = \begin{pmatrix}1 & 0 & \cdots & 0\\ 0 & 1 & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & 0 & \cdots & 1\end{pmatrix} \bigg| \begin{pmatrix}x_1\\x_2\\\vdots\\x_n\end{pmatrix}")

        with st.expander(T["gj_what_is_rref"], expanded=False):
            st.markdown(T["gj_rref_explain"])

        # ── Solution ─────────────────────────────────────────────────
        fig_sol_obj = None
        if solution is not None:
            st.subheader(T["gj_solution"])
            sol_cols = st.columns(min(n_size, 6))
            for i, xi in enumerate(solution):
                sol_cols[i % len(sol_cols)].metric(f"x{i+1}", smart_fmt(xi))

            # LaTeX solution
            sol_latex = r"x = \begin{pmatrix}" + r"\\".join(smart_fmt(xi) for xi in solution) + r"\end{pmatrix}"
            st.latex(sol_latex)

            # ── Verification ────────────────────────────────────────
            st.subheader(T["gj_verification"])
            Ax = A_input @ solution
            residual = np.linalg.norm(Ax - b_input)

            ver_col1, ver_col2 = st.columns([3, 1])
            with ver_col1:
                ver_data = {
                    "équation": [f"Équation {i+1}" for i in range(n_size)],
                    T["gj_ax"]: [smart_fmt(Ax[i]) for i in range(n_size)],
                    T["gj_bi"]: [smart_fmt(b_input[i]) for i in range(n_size)],
                    T["gj_diff"]: [f"{abs(Ax[i]-b_input[i]):.2e}" for i in range(n_size)],
                }
                st.dataframe(pd.DataFrame(ver_data), use_container_width=True)

            with ver_col2:
                st.metric(T["gj_residual"], f"{residual:.4e}")
                if residual < 1e-10:
                    st.success(T["gj_residual_negligible"])
                elif residual < 1e-6:
                    st.info(T["gj_residual_low"])
                else:
                    st.warning(T["gj_residual_high"])

            with st.expander(T["gj_what_is_residual"], expanded=False):
                st.markdown(T["gj_residual_explain"])

            # ── Bar chart of solution ────────────────────────────────
            fig_sol, ax_sol = plt.subplots(figsize=(max(6, n_size*1.2), 4))
            fig_sol.patch.set_facecolor("#0d1b2a")
            ax_sol.set_facecolor("#0d1b2a")
            colors = ["#4a90d9" if v >= 0 else "#f5a623" for v in solution]
            bars = ax_sol.bar([f"x{i+1}" for i in range(n_size)], solution, color=colors, edgecolor="#334", linewidth=1.2)
            y_range = max(abs(solution)) if max(abs(solution)) > 0 else 1
            for bar, val in zip(bars, solution):
                ax_sol.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + y_range * 0.03,
                            smart_fmt(val),
                            ha='center', va='bottom', color="#e2e8f0",
                            fontfamily="monospace", fontsize=9)
            ax_sol.axhline(0, color="#555", linewidth=0.8)
            ax_sol.set_xlabel("Variables", color="#aac")
            ax_sol.set_ylabel("Valeur", color="#aac")
            ax_sol.set_title("Solution x du système Ax = b", color="#e2e8f0", fontfamily="monospace")
            ax_sol.tick_params(colors="#aac")
            for spine in ax_sol.spines.values():
                spine.set_edgecolor("#334")
            ax_sol.grid(True, axis='y', linestyle="--", alpha=0.18, color="#4a90d9")
            st.pyplot(fig_sol)
            fig_sol_obj = fig_sol

        # ── Summary table ────────────────────────────────────────────
        st.divider()
        st.subheader(T["gj_summary"])
        synth = {
            "Indicateur": [T["gj_rank"], T["gj_det"], T["gj_cond"], "Nombre d'étapes",
                           T["gj_residual"]],
            "Valeur": [
                f"{rank} / {n_size}",
                smart_fmt(det) if det is not None else "N/A",
                f"{cond:.4e}" if cond is not None else "N/A",
                str(len(steps)),
                f"{residual:.4e}" if solution is not None else "N/A",
            ]
        }
        st.dataframe(pd.DataFrame(synth), use_container_width=True)

        # ── Downloads ─────────────────────────────────────────────────
        st.divider()
        st.subheader(T["download_title"])
        dl1, dl2, dl3 = st.columns(3)

        txt_gj = make_txt_gj(n_size, A_input, b_input, solution, steps, det, rank, cond)
        dl1.download_button(T["download_txt"], data=txt_gj,
                            file_name="gauss_jordan_report.txt", mime="text/plain")

        csv_gj = make_csv_gj(solution, A_input, b_input)
        dl2.download_button(T["download_csv"], data=csv_gj,
                            file_name="gauss_jordan_solution.csv", mime="text/csv")

        pdf_gj = make_pdf_gj(n_size, A_input, b_input, solution, steps, det, rank, cond, rref, fig_sol_obj)
        if pdf_gj:
            dl3.download_button(T["download_pdf"], data=pdf_gj,
                                file_name="gauss_jordan_report.pdf", mime="application/pdf")
        else:
            dl3.info("PDF: `pip install fpdf2`")


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: SIMPSON 1/3
# ═══════════════════════════════════════════════════════════════════════
elif st.session_state["page"] == "simp":

    with st.sidebar:
        st.subheader(T["params"])

        func_option = st.radio(T["function_choice"], [T["predefined"], T["custom"]])
        PRESETS = [
            "x**2", "x**3 - 3*x + 1", "1/x", "sqrt(x)", "abs(x)",
            "sin(x)", "cos(x)", "tan(x)", "arctan(x)", "arcsin(x)", "arccos(x)",
            "exp(x)", "exp(-x**2)", "log(x)", "sh(x)", "ch(x)", "th(x)",
            "argsh(x)", "argch(x)", "argth(x)", "1/(1+x**2)"
        ]
        if func_option == T["predefined"]:
            func_str = st.selectbox(T["select_function"], PRESETS)
        else:
            if "custom_func" not in st.session_state:
                st.session_state["custom_func"] = "x**2 * sin(x)"

            def add_sym(sym):
                st.session_state["custom_func"] += sym
            def clear_func():
                st.session_state["custom_func"] = ""
            def del_last():
                st.session_state["custom_func"] = st.session_state["custom_func"][:-1]

            func_str = st.text_input(T["enter_function"], key="custom_func")
            st.markdown("<small>🖩 **Clavier rapide :**</small>", unsafe_allow_html=True)
            pad_4cols = [
                [('7','7'),('8','8'),('9','9'),('/','/')],
                [('4','4'),('5','5'),('6','6'),('*','*')],
                [('1','1'),('2','2'),('3','3'),('-','-')],
                [('0','0'),('.','.'),('^','**'),('+','+')],
                [('(','('),(')',')'),('x','x'),('pi','pi')]
            ]
            for r_idx, row in enumerate(pad_4cols):
                cols = st.columns(4)
                for i, (label, val) in enumerate(row):
                    cols[i].button(label, on_click=add_sym, args=(val,), use_container_width=True, key=f"num_{r_idx}_{i}_{label}")
            pad_3cols = [
                [('sin','sin('),('cos','cos('),('tan','tan(')],
                [('exp','exp('),('log','log('),('sqrt','sqrt(')],
                [('asin','arcsin('),('acos','arccos('),('atan','arctan(')]
            ]
            for r_idx, row in enumerate(pad_3cols):
                cols = st.columns(3)
                for i, (label, val) in enumerate(row):
                    cols[i].button(label, on_click=add_sym, args=(val,), use_container_width=True, key=f"func_{r_idx}_{i}_{label}")
            c1, c2 = st.columns(2)
            c1.button("⌫", on_click=del_last, use_container_width=True, key="btn_del")
            c2.button("C", on_click=clear_func, use_container_width=True, key="btn_clear")
            st.caption(T["function_hint"])

        col_a, col_b = st.columns(2)
        a = col_a.number_input(T["lower"], value=0.0, step=0.5, format="%g")
        b = col_b.number_input(T["upper"], value=2.0, step=0.5, format="%g")

        n_input = st.number_input(T["intervals"], min_value=2, value=6, step=2)
        n = int(n_input)
        n_adjusted = False
        if n % 2 != 0:
            n += 1
            n_adjusted = True

        st.divider()
        show_plot     = st.checkbox(T["show_plot"], value=True)
        show_table    = st.checkbox(T["show_table"], value=False)
        compare_exact = st.checkbox(T["compare_exact"], value=True)
        show_conv     = st.checkbox(T["convergence"], value=False)

        st.divider()
        run = st.button(T["calculate"], type="primary", use_container_width=True)

    with st.expander(T["formula_title"], expanded=False):
        st.latex(r"\int_a^b f(x)\,dx \approx \frac{h}{3}\left[f(x_0)+4\sum_{i\text{ impair}}f(x_i)+2\sum_{i\text{ pair}}f(x_i)+f(x_n)\right]")
        st.markdown("avec $h = (b-a)/n$, $n$ pair")

    if run:
        if n_adjusted:
            st.warning(T["n_must_even"])

        try:
            test_x = np.linspace(a, b, 5)
            evaluate(func_str, test_x)
        except Exception as e:
            st.error(f"{T['error_func']} : {e}")
            st.stop()

        exact_val = None
        if compare_exact:
            exact_val = exact_integral(func_str, a, b)
            if exact_val is None:
                st.warning(T["error_exact"])

        x_vals, y_vals, result, fig = None, None, None, None
        try:
            x_vals, y_vals, result, n = simpson_13(func_str, a, b, n)
        except Exception as e:
            st.error(f"{T['error_calc']} : {e}")
            st.stop()

        st.subheader(T["results"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(T["numerical_result"], f"{result:.10f}")
        if exact_val is not None:
            abs_err = abs(result - exact_val)
            rel_err = abs_err / abs(exact_val) if exact_val != 0 else float("inf")
            c2.metric(T["exact_value"], f"{exact_val:.10f}")
            c3.metric(T["abs_error"], f"{abs_err:.3e}")
            c4.metric(T["rel_error"], f"{rel_err:.3e}")

        if show_plot:
            fig, ax = plt.subplots(figsize=(11, 4.5))
            fig.patch.set_facecolor("#0d1b2a")
            ax.set_facecolor("#0d1b2a")
            x_fine = np.linspace(a, b, 600)
            y_fine = evaluate(func_str, x_fine)
            ax.plot(x_fine, y_fine, color="#4a90d9", linewidth=2.2, label=f"f(x) = {func_str}", zorder=4)
            ax.fill_between(x_fine, y_fine, alpha=0.15, color="#4a90d9")
            for i in range(len(x_vals) - 1):
                ax.fill_between([x_vals[i], x_vals[i+1]], [y_vals[i], y_vals[i+1]], alpha=0.35,
                                color="#f5a623", zorder=2)
                ax.plot([x_vals[i], x_vals[i]], [0, y_vals[i]], color="#f5a623", linewidth=0.7,
                        linestyle="--", zorder=3)
            ax.plot(x_vals, y_vals, 'o', color="#f5a623", markersize=5, label="Simpson 1/3", zorder=5)
            ax.axhline(0, color="#555", linewidth=0.8)
            ax.set_xlabel("x", color="#aac")
            ax.set_ylabel("f(x)", color="#aac")
            ax.set_title(f"∫ f(x) dx ≈ {result:.8f}", color="#e2e8f0", fontfamily="monospace", fontsize=12)
            ax.tick_params(colors="#aac")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334")
            ax.legend(facecolor="#0d1b2a", labelcolor="#e2e8f0", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.18, color="#4a90d9")
            st.pyplot(fig)

        if show_conv and exact_val is not None:
            ns_c, errs_c = convergence_data(func_str, a, b, exact_val)
            fig_c, ax_c = plt.subplots(figsize=(9, 3.5))
            fig_c.patch.set_facecolor("#0d1b2a")
            ax_c.set_facecolor("#0d1b2a")
            valid = [(n2, e) for n2, e in zip(ns_c, errs_c) if e > 0 and not np.isnan(e)]
            if valid:
                ns_v, es_v = zip(*valid)
                ax_c.loglog(ns_v, es_v, 'o-', color="#4ef0c4", linewidth=2, markersize=4)
            ax_c.set_xlabel(T["conv_n"], color="#aac")
            ax_c.set_ylabel(T["conv_err"], color="#aac")
            ax_c.set_title(T["conv_title"], color="#e2e8f0", fontfamily="monospace")
            ax_c.tick_params(colors="#aac")
            for spine in ax_c.spines.values():
                spine.set_edgecolor("#334")
            ax_c.grid(True, linestyle="--", alpha=0.2, color="#4ef0c4")
            st.pyplot(fig_c)

        if show_table:
            st.subheader(T["eval_points"])
            df = pd.DataFrame({"i": range(len(x_vals)), "x_i": x_vals, "f(x_i)": y_vals})
            st.dataframe(df.style.format({"x_i": "{:.8f}", "f(x_i)": "{:.8f}"}), use_container_width=True)

        st.divider()
        st.subheader(T["download_title"])
        dl1, dl2, dl3 = st.columns(3)
        txt_bytes = make_txt_simp(func_str, a, b, n, result, exact_val, x_vals, y_vals)
        dl1.download_button(T["download_txt"], data=txt_bytes,
                            file_name="simpson_report.txt", mime="text/plain")
        csv_bytes = make_csv_simp(x_vals, y_vals)
        dl2.download_button(T["download_csv"], data=csv_bytes,
                            file_name="simpson_points.csv", mime="text/csv")
        pdf_bytes = make_pdf_simp(func_str, a, b, n, result, exact_val, fig if show_plot else None)
        if pdf_bytes:
            dl3.download_button(T["download_pdf"], data=pdf_bytes,
                                file_name="simpson_report.pdf", mime="application/pdf")
        else:
            dl3.info("PDF: `pip install fpdf2`")

    st.divider()
    with st.expander(T["about_title"], expanded=False):
        st.markdown(T["about_simp"])
        st.markdown("""
---
**Complexité / Complexity:**

| Méthode | Ordre erreur | Contrainte |
|---------|-------------|-----------|
| Simpson 1/3 | O(h⁴) | n pair |

> **Astuce:** Simpson 1/3 nécessite ~10× moins de sous-intervalles que les trapèzes pour la même précision.
        """)