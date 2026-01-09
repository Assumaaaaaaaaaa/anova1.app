import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Projet ANOVA – Finance", layout="wide")

# ============================================
# NOUVELLE INTERFACE SIMPLIFIÉE
# ============================================
st.markdown("""
<div style="text-align: center;">
    <h1 style="color: #1E3A8A; margin-bottom: 10px;">📊 Analyse ANOVA des Portefeuilles Financiers</h1>
    <p style="color: #4B5563; font-size: 16px; max-width: 800px; margin: 0 auto;">
        Application interactive pour comparer la performance des portefeuilles selon le type, 
        la stratégie de gestion et la zone géographique
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# Sidebar – Import données
# =========================
st.sidebar.header("📂 Importation des données")
file = st.sidebar.file_uploader(
    "Importer un fichier CSV",
    type=["csv"],
    help="Colonnes requises : Rendement, Portefeuille, Strategie, Zone"
)

# =========================
# Fonction format p-value
# =========================
def format_pvalue(p):
    if p < 0.001:
        return "< 0.001"
    else:
        return f"{p:.4f}"

# =========================
# Application principale
# =========================
if file is not None:
    data = pd.read_csv(file)
    
    # Vérification des colonnes
    required_cols = ['Rendement', 'Portefeuille', 'Strategie', 'Zone']
    if all(col in data.columns for col in required_cols):
        st.success("✅ Données chargées avec succès")
        
        tabs = st.tabs([
            "📘 Cours ANOVA",
            "📂 Données",
            "🧪 ANOVA 1 facteur",
            "🧪 ANOVA 2 facteurs",
            "✅ Validation",
            "💰 Interprétation financière"
        ])

        # =====================================================
        # TAB 1 – COURS ANOVA
        # =====================================================
        with tabs[0]:
            st.header("Analyse de la variance (ANOVA) – Cours")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("1. Principe général")
                st.markdown("""
                L'ANOVA permet de comparer les moyennes de plusieurs groupes 
                pour déterminer si au moins l'une d'entre elles diffère 
                significativement des autres.
                """)
                
                st.subheader("2. Hypothèses statistiques")
                st.markdown("**H₀ :** μ₁ = μ₂ = ... = μₖ")
                st.markdown("**H₁ :** Au moins un μᵢ ≠ μⱼ")
                
                st.subheader("3. Types d'ANOVA")
                st.markdown("""
                - ANOVA à un facteur
                - ANOVA à deux facteurs
                - ANOVA avec interaction
                """)
            
            with col2:
                st.subheader("4. Décomposition de la variance")
                st.latex(r"SST = SSB + SSW")
                st.markdown("""
                - **SST** : Variance totale
                - **SSB** : Variance entre groupes
                - **SSW** : Variance à l'intérieur des groupes
                """)
                
                st.subheader("5. Statistique F")
                st.latex(r"F = \frac{MSB}{MSW}")
                st.markdown("""
                - **MSB** = SSB / (k-1)
                - **MSW** = SSW / (n-k)
                """)
                
                st.subheader("6. Conditions de validité")
                st.markdown("""
                1. Normalité des résidus
                2. Homoscédasticité
                3. Indépendance des observations
                """)

        # =====================================================
        # TAB 2 – DONNÉES
        # =====================================================
        with tabs[1]:
            st.header("Exploration des données")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Aperçu des données")
                st.dataframe(data.head(), use_container_width=True)
                st.caption(f"Total : {len(data)} observations")
            
            with col2:
                st.subheader("Informations générales")
                st.write(f"**Variables :** {len(data.columns)}")
                st.write(f"**Observations :** {len(data)}")
                st.write("**Colonnes :**")
                for col in data.columns:
                    st.write(f"- {col}")
            
            st.subheader("Statistiques descriptives")
            st.dataframe(data.describe().round(3))

        # =====================================================
        # TAB 3 – ANOVA 1 FACTEUR
        # =====================================================
        with tabs[2]:
            st.header("ANOVA à un facteur : Type de portefeuille")
            
            st.markdown("**Question :** Les rendements moyens diffèrent-ils selon le type de portefeuille ?")
            
            # Calcul ANOVA
            model1 = ols('Rendement ~ C(Portefeuille)', data=data).fit()
            anova1 = sm.stats.anova_lm(model1, typ=2)
            
            # Affichage tableau
            st.subheader("Table ANOVA")
            anova1_display = anova1.copy()
            anova1_display['PR(>F)'] = anova1['PR(>F)'].apply(format_pvalue)
            st.dataframe(anova1_display.style.format({'sum_sq': '{:.3f}', 'F': '{:.3f}'}))
            
            # Interprétation
            pval1 = anova1['PR(>F)'][0]
            
            st.subheader("Interprétation")
            if pval1 < 0.001:
                st.error("**p-value < 0.001** → différence très significative entre portefeuilles")
            elif pval1 < 0.05:
                st.warning("**p-value < 0.05** → différence significative entre portefeuilles")
            else:
                st.success("**p-value ≥ 0.05** → pas de différence significative")
            
            # Conclusion financière
            st.subheader("Conclusion financière")
            if pval1 < 0.05:
                meilleur = data.groupby('Portefeuille')['Rendement'].mean().idxmax()
                rendement = data.groupby('Portefeuille')['Rendement'].mean().max()
                st.info(f"Le portefeuille **{meilleur}** présente le rendement moyen le plus élevé ({rendement:.2f}%).")
            else:
                st.info("Aucun type de portefeuille ne se distingue significativement en termes de rendement moyen.")

        # =====================================================
        # TAB 4 – ANOVA 2 FACTEURS
        # =====================================================
        with tabs[3]:
            st.header("ANOVA à deux facteurs avec interaction")
            
            st.markdown("**Question :** La stratégie de gestion influence-t-elle les rendements indépendamment du type de portefeuille ?")
            
            # Calcul ANOVA
            model2 = ols('Rendement ~ C(Portefeuille) * C(Strategie)', data=data).fit()
            anova2 = sm.stats.anova_lm(model2, typ=2)
            
            # Affichage tableau
            st.subheader("Table ANOVA")
            anova2_display = anova2.copy()
            anova2_display['PR(>F)'] = anova2['PR(>F)'].apply(format_pvalue)
            st.dataframe(anova2_display.style.format({'sum_sq': '{:.3f}', 'F': '{:.3f}'}))
            
            # Effets principaux
            st.subheader("Analyse des effets")
            
            p_portefeuille = anova2.loc['C(Portefeuille)', 'PR(>F)'] if 'C(Portefeuille)' in anova2.index else 1.0
            p_strategie = anova2.loc['C(Strategie)', 'PR(>F)'] if 'C(Strategie)' in anova2.index else 1.0
            p_interaction = anova2.loc['C(Portefeuille):C(Strategie)', 'PR(>F)'] if 'C(Portefeuille):C(Strategie)' in anova2.index else 1.0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Effet Portefeuille", 
                         "✓" if p_portefeuille < 0.05 else "✗",
                         delta=f"p={format_pvalue(p_portefeuille)}",
                         delta_color="normal" if p_portefeuille >= 0.05 else "inverse")
            
            with col2:
                st.metric("Effet Stratégie",
                         "✓" if p_strategie < 0.05 else "✗",
                         delta=f"p={format_pvalue(p_strategie)}",
                         delta_color="normal" if p_strategie >= 0.05 else "inverse")
            
            with col3:
                st.metric("Interaction",
                         "✓" if p_interaction < 0.05 else "✗",
                         delta=f"p={format_pvalue(p_interaction)}",
                         delta_color="normal" if p_interaction >= 0.05 else "inverse")
            
            # NOUVELLE PRÉSENTATION DU TEST TUKEY
            st.subheader("3. Tests post-hoc de Tukey (Stratégie)")
            
            # Calcul du test Tukey
            tukey_result = pairwise_tukeyhsd(data['Rendement'], data['Strategie'])
            
            # Création d'un cadre stylisé pour afficher les résultats
            st.markdown("""
            <div style="
                background-color: #f8f9fa;
                border-left: 4px solid #1E3A8A;
                padding: 15px;
                margin: 10px 0;
                border-radius: 4px;
            ">
            """, unsafe_allow_html=True)
            
            # Affichage sous forme de métriques
            active_mean = data[data['Strategie'] == 'Active']['Rendement'].mean() if 'Active' in data['Strategie'].values else 0
            passive_mean = data[data['Strategie'] == 'Passive']['Rendement'].mean() if 'Passive' in data['Strategie'].values else 0
            difference = active_mean - passive_mean
            
            col_t1, col_t2, col_t3 = st.columns(3)
            
            with col_t1:
                st.metric("Active", f"{active_mean:.3f}%")
            
            with col_t2:
                st.metric("Passive", f"{passive_mean:.3f}%")
            
            with col_t3:
                st.metric("Différence", f"{difference:.3f}%",
                         delta_color="inverse" if difference < 0 else "normal")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Résultats détaillés du test
            with st.expander("Voir les résultats détaillés du test Tukey"):
                st.text(tukey_result.summary())
            
            # Interprétation
            st.markdown("**Interprétation :**")
            if hasattr(tukey_result, 'reject') and len(tukey_result.reject) > 0:
                if tukey_result.reject[0]:
                    if difference > 0:
                        st.success(f"✅ La stratégie **Active** surperforme significativement la stratégie Passive")
                    else:
                        st.success(f"✅ La stratégie **Passive** surperforme significativement la stratégie Active")
                else:
                    st.info("ℹ️ Pas de différence significative entre les stratégies Active et Passive")
            
            # Interaction
            st.subheader("Effet d'interaction")
            if p_interaction < 0.05:
                st.warning("L'effet de la stratégie dépend du type de portefeuille")
            else:
                st.info("L'effet de la stratégie est similaire pour tous les types de portefeuilles")

        # =====================================================
        # TAB 5 – VALIDATION
        # =====================================================
        with tabs[4]:
            st.header("Validation des hypothèses")
            
            residuals = model2.resid
            
            col_val1, col_val2 = st.columns(2)
            
            with col_val1:
                st.subheader("Test de Shapiro-Wilk")
                stat, p_shapiro = stats.shapiro(residuals)
                st.write(f"Statistique : {stat:.4f}")
                st.write(f"p-value : {format_pvalue(p_shapiro)}")
                
                if p_shapiro >= 0.05:
                    st.success("✅ Normalité acceptée")
                else:
                    st.warning("⚠️ Normalité rejetée")
            
            with col_val2:
                st.subheader("Test de Levene")
                groups = [data['Rendement'][data['Portefeuille'] == g] for g in data['Portefeuille'].unique()]
                stat, p_levene = stats.levene(*groups)
                st.write(f"Statistique : {stat:.4f}")
                st.write(f"p-value : {format_pvalue(p_levene)}")
                
                if p_levene >= 0.05:
                    st.success("✅ Homoscédasticité acceptée")
                else:
                    st.warning("⚠️ Homoscédasticité rejetée")
            
            st.subheader("Analyse graphique des résidus")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(residuals, kde=True, ax=ax)
            ax.set_title('Distribution des résidus')
            ax.set_xlabel('Résidus')
            ax.set_ylabel('Densité')
            st.pyplot(fig)

        # =====================================================
        # TAB 6 – INTERPRÉTATION FINANCIÈRE
        # =====================================================
        with tabs[5]:
            st.header("💰 Interprétation financière")
            
            # Calcul des performances
            perf_portefeuille = data.groupby('Portefeuille')['Rendement'].mean()
            perf_strategie = data.groupby('Strategie')['Rendement'].mean()
            
            st.subheader("1. Performance par stratégie")
            col_f1, col_f2 = st.columns(2)
            
            with col_f1:
                st.write("**Rendements moyens :**")
                for strategie, moyenne in perf_strategie.items():
                    st.write(f"- {strategie} : {moyenne:.2f}%")
            
            with col_f2:
                meilleure_strategie = perf_strategie.idxmax() if len(perf_strategie) > 0 else "N/A"
                if 'p_strategie' in locals() and p_strategie < 0.05:
                    st.success(f"**{meilleure_strategie}** est la plus performante")
                else:
                    st.info("Pas de stratégie significativement plus performante")
            
            st.subheader("2. Surperformance confirmée ?")
            if pval1 < 0.05 or ('p_strategie' in locals() and p_strategie < 0.05):
                st.success("✅ Oui, surperformance statistique détectée")
            else:
                st.warning("❌ Non, pas de surperformance statistique")
            
            st.subheader("3. Recommandations")
            if pval1 < 0.05:
                meilleur_port = perf_portefeuille.idxmax()
                st.info(f"Privilégier les portefeuilles **{meilleur_port}**")
            
            if 'p_strategie' in locals() and p_strategie < 0.05:
                st.info(f"Adopter la stratégie **{meilleure_strategie}**")
            
            st.subheader("4. Limites de l'ANOVA")
            st.markdown("""
            - Hypothèse de normalité souvent violée en finance
            - Ne tient pas compte du risque (volatilité)
            - Relations linéaires uniquement
            - Ignore l'autocorrélation temporelle
            """)

    else:
        st.error(f"❌ Colonnes requises manquantes. Assurez-vous que votre fichier contient : {', '.join(required_cols)}")
        st.info("""
        **Structure du fichier attendue :**
        ```
        Rendement,Portefeuille,Strategie,Zone
        1.2,Actions,Active,USA
        0.7,Obligations,Passive,Europe
        1.8,Mixte,Active,Afrique
        ```
        """)

else:
    # ============================================
    # PAGE D'ACCUEIL SIMPLIFIÉE
    # ============================================
    st.markdown("""
    <div style="
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin: 30px 0;
    ">
        <h3 style="color: #1E3A8A;">Bienvenue dans l'analyseur ANOVA</h3>
        <p style="color: #4B5563;">
            Importez vos données financières pour analyser la performance des portefeuilles
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("📤 Importez un fichier CSV dans la barre latérale pour commencer l'analyse")
    
    # Instructions
    with st.expander("📋 Instructions et exigences"):
        st.markdown("""
        **Structure requise du fichier CSV :**
        
        | Rendement | Portefeuille | Strategie | Zone |
        |-----------|--------------|-----------|------|
        | 1.2 | Actions | Active | USA |
        | 0.7 | Obligations | Passive | Europe |
        | 1.8 | Mixte | Active | Afrique |
        
        **Exigences :**
        - Format : CSV (séparateur virgule)
        - Colonnes exactes : Rendement, Portefeuille, Strategie, Zone
        - **Rendement** : valeurs numériques (rendements mensuels en %)
        - **Portefeuille** : "Actions", "Obligations" ou "Mixte"
        - **Strategie** : "Active" ou "Passive"
        - **Zone** : "Afrique", "Europe" ou "USA"
        
        **Analyses disponibles :**
        1. ANOVA à un facteur (Type de portefeuille)
        2. ANOVA à deux facteurs avec interaction
        3. Tests post-hoc de Tukey
        4. Validation des hypothèses statistiques
        5. Interprétation financière
        """)

