# Journal des erreurs (Partie 6 de la présentation)

La présentation doit montrer **au moins deux difficultés réellement rencontrées**.
Notez chaque erreur **au moment où elle arrive** : message complet, capture d'écran, puis la correction.

## Modèle d'entrée

### [Date] — [Titre court]
1. **Problème :** …
2. **Message / comportement observé :**
   ```
   (coller le log ou l'erreur complète ; capture dans docs/img/)
   ```
3. **Causes possibles :** …
4. **Solutions envisagées → solution retenue :** …
5. **Résultat après correction :** (capture, valeur avant/après)

---

### 2026-11-11 — Annotations de test chargées à la place de celles d'entraînement
1. **Problème :** la cellule de détection du dataset (notebook Kaggle, section 2) a sélectionné le dossier
   d'annotations **de test** au lieu de celui d'entraînement.
2. **Comportement observé :**
   ```
   XML    : /kaggle/input/datasets/bratjay/ua-detrac-orig/DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML (40 séquences)
   IMAGES : /kaggle/input/datasets/bratjay/ua-detrac-orig/DETRAC-Images/DETRAC-Images (40 séquences avec images)
   ```
   Attendu : 60 séquences d'entraînement. Pas d'erreur Python : le bug était silencieux.
   (capture : `docs/img/err_xml_test.png`)
3. **Cause :** `XML = os.path.dirname(sorted(xmls)[0])` prend le premier chemin dans l'ordre alphabétique ;
   or `DETRAC-Test-…` < `DETRAC-Train-…` (« e » < « r »). Le code supposait une version du dataset ne
   contenant que les annotations d'entraînement.
4. **Solutions envisagées :** (a) chemin écrit en dur → fragile si on change de version du dataset ;
   (b) **retenue :** lister tous les dossiers XML et choisir explicitement celui dont le nom contient « Train »,
   avec une assertion explicite sinon, et afficher tous les dossiers trouvés.
5. **Résultat après correction :** les deux dossiers sont listés, le bon est choisi :
   ```
   XML    : /kaggle/input/datasets/bratjay/ua-detrac-orig/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML (60 séquences)
   IMAGES : /kaggle/input/datasets/bratjay/ua-detrac-orig/DETRAC-Images/DETRAC-Images (60 séquences avec images)
   ```
   60 séquences = valeur attendue d'après l'article. (capture : `docs/img/fix_xml_train.png`)

**Leçon :** vérifier les comptes (60 train / 40 test d'après l'article UA-DETRAC) avant de lancer un traitement long.

---

### 2026-11-11 — Échec de l'évaluation TrackEval après 2 h de calcul
1. **Problème :** la version 1 du notebook Kaggle a échoué au bout de **7595 s**, à la toute dernière étape
   (section 6). L'entraînement (30 époques) et le suivi des 4 configurations sur les 9 séquences de validation
   avaient pourtant réussi.
2. **Message observé :**
   ```
   File "/kaggle/working/TrackEval/trackeval/datasets/mot_challenge_2d_box.py", line 156, in _get_seq_info
       if not os.path.isfile(seqmap_file):
   TypeError: stat: path should be string, bytes, os.PathLike or integer, not list
   ```
   puis, dans la cellule suivante (erreur en cascade) :
   ```
   ValueError: No objects to concatenate
   ```
   La configuration affichée par TrackEval montre la cause : `SEQMAP_FILE : ['/kaggle/working/trackeval/seqmap.txt']`
   (une liste au lieu d'une chaîne). (capture : `docs/img/err_trackeval_seqmap.png`)
3. **Cause :** le script en ligne de commande `run_mot_challenge.py` déclare tous ses arguments avec `nargs='+'`
   et ne reconvertit en chaîne que les paramètres dont la valeur par défaut est une chaîne. Or la valeur par
   défaut de `SEQMAP_FILE` est `None` → la liste `[chemin]` est transmise telle quelle à `os.path.isfile`.
   La 2ᵉ erreur n'est qu'une conséquence : aucun fichier `pedestrian_summary.txt` n'a été produit.
4. **Solutions envisagées :** (a) passer par `--SEQMAP_FOLDER` + `--BENCHMARK` pour contourner l'argument ;
   (b) modifier le script de TrackEval ; (c) **retenue :** appeler TrackEval par son **API Python**
   (`trackeval.Evaluator(...).evaluate(...)`) avec un dictionnaire de configuration : plus d'analyse
   d'arguments, et les métriques sont lues directement dans le résultat au lieu d'un fichier texte.
   En plus : correctif préventif `np.float → float` (numpy ≥ 1.24), et **reprise** depuis l'Output de la
   version 1 (poids `best.pt` + sorties de suivi) pour ne pas refaire 2 h de calcul.
5. **Résultat après correction :** la version 2 s'exécute en **553 s** (au lieu de 7595 s) grâce à la reprise,
   et produit le tableau attendu (9 séquences de validation) :
   ```
   config               HOTA  MOTA  IDF1  IDSW
   ft_botsort_tuned     61.6  42.8  75.8   129
   baseline_bytetrack   57.9  48.6  73.0   359
   ft_bytetrack_tuned   57.3  35.4  72.6   230
   ft_bytetrack         56.9  32.6  71.9   245
   ```
   (capture : `docs/img/fix_trackeval.png`)

**Leçon :** tester la chaîne d'évaluation sur **une seule séquence** avant le long calcul, et rendre le notebook
reprenable (ne recalculer que ce qui manque).

---

### 2026-11-11 — Le modèle fine-tuné a un MOTA plus faible que le modèle COCO
1. **Problème :** après fine-tuning, le MOTA **baisse** (48.6 → 32.6 avec ByteTrack), alors que le HOTA et
   l'IDF1 de la meilleure configuration augmentent. Sur la séquence MVI_20035, le MOTA devient même **négatif**.
2. **Comportement observé** (TrackEval, 9 séquences de validation, `pedestrian_detailed.csv`) :
   ```
   config               MOTA   FP      FN
   baseline_bytetrack   48.6   33228   19628
   ft_bytetrack         32.6   60724    8828
   ft_botsort_tuned     42.8   54468    4652
   MVI_20035 (ft_bytetrack) : MOTA = -0.101, FP = 12891, FN = 159
   ```
   Le modèle fine-tuné rate 4× moins de véhicules (FN) mais a presque 2× plus de faux positifs (FP).
3. **Causes possibles :** (a) seuil de confiance trop bas ; (b) détections en double ;
   (c) **retenue après analyse :** UA-DETRAC n'annote pas les véhicules des *ignored regions* (route au loin).
   Le modèle fine-tuné, entraîné sur ces caméras, voit ces petits véhicules lointains ; ils n'ont pas de
   vérité terrain et comptent donc comme FP. Le modèle COCO les voit moins → moins de FP.
   Le MOTA (`1 − (FN + FP + IDSW) / GT`) est très sensible aux FP, contrairement au HOTA.
4. **Solution retenue :** appliquer le protocole officiel d'UA-DETRAC : retirer, dans la vérité terrain et
   dans les sorties des trackers, les boîtes couvertes à plus de 50 % par les zones ignorées
   (`scripts/filter_ignored.py`, zones exportées par `convert_detrac.py --mot-gt` dans `ignored.txt`),
   puis réévaluer. Les deux protocoles (brut / sans zones ignorées) sont présentés côte à côte.
5. **Résultat après correction :** (à compléter avec le tableau de la version 3)

**Leçon :** une métrique qui baisse n'est pas forcément un modèle qui régresse ; regarder FP et FN séparément
et vérifier que le protocole d'évaluation correspond à celui du dataset.

---

## Points à surveiller (problèmes fréquents sur ce type de projet)

- Session Kaggle interrompue pendant l'entraînement → reprendre avec `last.pt` (`resume=True`).
- `CUDA out of memory` → réduire `batch` ou `imgsz`.
- TrackEval avec un numpy récent : `AttributeError: module 'numpy' has no attribute 'float'`.
- Labels décalés ou boîtes hors image après conversion → cellule de contrôle visuel du notebook.
- Confusion BGR/RGB à l'affichage (OpenCV lit en BGR).
- Vidéo `mp4v` non lisible dans le navigateur (démo Gradio) → ré-encodage H.264.
- Changements d'identifiant (ID switches) quand un bus cache une voiture → `track_buffer`.
- Pas d'accès Internet ni GPU sur Kaggle → compte non vérifié par téléphone.
