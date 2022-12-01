# The Airbnb Data Science Case

This repo holds the notebooks to investigate the Airbnb dataset from see [here](http://insideairbnb.com/)

It's purpose is to teach data science fundamentals.

If you worked with it and have suggestions, please contribute your improvements.

## install

Use pip to install the requirements:

```
pip install -r requirements.txt
```

The plotting library **cartopy** has external dependencies see [here](https://scitools.org.uk/cartopy/docs/latest/installing.html)

# Usage

Solution scripts, can be run without any further modifications.

They contain comments questions and the solutions.

From the solution scripts, exercise scripts are generated by running:

```
python create_exercise.py --ifiles solutions*.ipynb
```

This removes lines that are tagged at the end with: `# REMOVE`

and replaces them with the placeholder `###!!! INSERT SOLUTION`.

For an examples, have a look in the first notebook.

## Git

Before staging, remove all cell output.


## Overview:

Die angegebenen Inhalte sind ausschließlich für die Durchführung eines dreitägigen Data Science Essentials Workshop auf Basis der Programmiersprache Python vom 5. – 7.12.22. zur Verfügung gestellt. Der Kurs ist inhaltlich auf die Techniker Krankenkasse zugeschnitten und behandelt die Kernkomponenten eines erfolgreichen Data Science Projekts. Diese Inhalte gliedern sich in die erfolgskritischen Grundlagen für ganzheitliche Data Science Projekte (Tag 1), essenzielle Schritte der Datenvorbereitung und Modellierung (Tag 2) und aufbauende Modellierungs- und Validierungsmethodiken (Tag 3). Ziel ist die Wissensvermittlung durch einen theoretischen Teil, gepaart mit praktischen Demonstrationen und Übungen zur Festigung des erlernten Wissens. Das Data Science Essentials Coaching bildet die Grundlage für weitere Themen wie fortgeschrittene Modellierungsansätze oder die operative Einbindung von Modellen im Unternehmenskontext. Diese Themen können in einem weiteren Data Science Advanced Coaching adressiert werden und bauen nahtlos auf dem Data Science Essentials Inhalten auf.

- Tag 1: Advanced Analytics in a Nutshell
    - Einführung Data Mining / Data Science
        - Datenstruktur / Zielgrößen
        - Übersicht Algorithmen

    - Weitere Modellierungsverfahren
        - Assoziationsanalyse
        - Segmentierung / Clustering
        - Anomalieerkennung

    - Vorgehen Data Science Projekt
        - Lifecycle (Selektion, Exploration, Modifkation, Modellierung, Validierung, Deployment)
        - Vorgehensmodell (KDD, Crisp-DM, SEMMA)

    - Data Science @ Work
        - Typische Ursachen für fehlgeschlagene Data Science Projekte
        - Prozessverständnis
        - Datenverständnis
        - Projektvorgehen – Vorgehen/Randpunkte

-  Tag 2: Case-Study + Datenvorbereitung / Regression
    - Case Study-Vorstellung (Use-Case wird in sämtlichen Demonstrationen und Übungen als Praxisbeispiel verwendet)
        - Vorstellung des Airbnb Use Cases sowie der korrespondierenden Datenbasis.
        - Setup (Cloud Pak 4 Data / Jupyter Notebook)

    - Datenvorverarbeitung und Regression
        - Lineare Regression (Loss Funktionen, R², Hypothesentests (Chi-Square)..)
        - Variablenselektion (Forward / Backward / Stepwise Selection)
        - Datenvorverarbeitung (Transformation, Binning, Dummy-Coding, Imputation, ..)
        - Logistische Regression (Herleitung, Logit-Transformation)

- Tag 3: Entscheidungsbäume und Modellvalidierung
    - Entscheidungsbäume und andere baumbasierte Algorithmen
        - Einführung (Regression vs. Klassifizierung, Vergleich zu anderen Algorithmen)
        - Grundlagen (Split-Search, Overfitting / Validierung (Pruning))
        - Ensemble-Modelle (Bootstrapping, Boosting, Modelle (Random Forest, Gradient Boosting)
        
    - Model Evaluation und -auswahl
        - Konzept / Rahmenbedingungen / Anwendungsszenarien
        - Variablenselektion
        - Honest Assessment – Kreuzvalidierung
        - Model Metriken (Accuracy, Sensitivity, Binär / Ranking / Regression)
        - Vergleich von Modellcharakteristiken (Interpretierbarkeit, Vorteiles, Drawbacks)

Notebooks: [Dr. Mares Barekzai](mailto:mbarekzai@positivethinking.tech)

Ansprechpartner: [Nicola Zäh](mailto:nzaeh@positivethinking.tech)

Beteiligte Teams:
    - 0959 Team Service-& Kontaktanalytik
    - 0918 Team Projekt TKalpha