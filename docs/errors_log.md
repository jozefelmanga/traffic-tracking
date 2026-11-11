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

## Points à surveiller (problèmes fréquents sur ce type de projet)

- Session Kaggle interrompue pendant l'entraînement → reprendre avec `last.pt` (`resume=True`).
- `CUDA out of memory` → réduire `batch` ou `imgsz`.
- TrackEval avec un numpy récent : `AttributeError: module 'numpy' has no attribute 'float'`.
- Labels décalés ou boîtes hors image après conversion → cellule de contrôle visuel du notebook.
- Confusion BGR/RGB à l'affichage (OpenCV lit en BGR).
- Vidéo `mp4v` non lisible dans le navigateur (démo Gradio) → ré-encodage H.264.
- Changements d'identifiant (ID switches) quand un bus cache une voiture → `track_buffer`.
- Pas d'accès Internet ni GPU sur Kaggle → compte non vérifié par téléphone.
