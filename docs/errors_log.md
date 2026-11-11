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

## Points à surveiller (problèmes fréquents sur ce type de projet)

- Session Kaggle interrompue pendant l'entraînement → reprendre avec `last.pt` (`resume=True`).
- `CUDA out of memory` → réduire `batch` ou `imgsz`.
- TrackEval avec un numpy récent : `AttributeError: module 'numpy' has no attribute 'float'`.
- Labels décalés ou boîtes hors image après conversion → cellule de contrôle visuel du notebook.
- Confusion BGR/RGB à l'affichage (OpenCV lit en BGR).
- Vidéo `mp4v` non lisible dans le navigateur (démo Gradio) → ré-encodage H.264.
- Changements d'identifiant (ID switches) quand un bus cache une voiture → `track_buffer`.
- Pas d'accès Internet ni GPU sur Kaggle → compte non vérifié par téléphone.
