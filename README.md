# Silhouette Blender

Addon Blender qui crée un plan dont chaque vertex est snappé sur la surface d'un objet cible via ray-cast vertical, ce qui produit une **silhouette + relief Z** propre.

Idéal pour :
- Extraire un bas-relief d'un mesh dense (BASIFY, sculpts, etc.) vers un plan déformé léger
- Créer une heightmap topologique d'un objet
- Générer un fond/socle qui épouse la forme d'un objet posé dessus

## Installation

1. `Edit > Preferences > Add-ons > Install`
2. Sélectionner `silhouette_plane_addon.py`
3. Cocher **Plan Silhouette** dans la liste
4. Sidebar 3D View (`N`) > onglet **Silhouette**

## Modes

L'addon propose **deux modes** :

### Ray-cast grille (défaut)
Génère une grille subdivisée et ray-caste chaque vert vers la cible. Donne une **topologie quad propre et régulière**, résolution réglable. Peut produire un effet escalier sur les bords (mitigé par les options de lissage).

### Extraire surface source
Copie directement les faces du dessus de la cible (celles dont la normale pointe vers le haut). **Fidélité parfaite, aucun escalier**, mais hérite de la topologie de la source (peut être très dense, parfois sale).

## Utilisation

1. Choisir le **mode** (Ray-cast ou Extraire)
2. Définir la **Cible** (mesh source)
3. Mode Ray-cast : régler **Résolution** et options de lissage
4. Mode Extraire : régler **Seuil normal Z** (0.1 par défaut capture toutes les faces "plutôt vers le haut")
5. Cliquer **Créer Plan Silhouette**

## Réglages

| Paramètre | Description |
|---|---|
| **Cible** | Mesh à ray-caster (obligatoire) |
| **Limites XY** | Objet définissant l'emprise X/Y du plan (défaut = bbox cible) |
| **Résolution X / Y** | Densité de la grille de départ |
| **Origine Z des rayons** | Hauteur de départ des rayons (doit être au-dessus de la cible) |
| **Distance max** | Longueur max d'un rayon |
| **Lisser les bordures** | Snap les verts de bord sur les contours réels de la cible (anti-escalier) |
| **Distance snap max** | Tolérance pour le snap des bordures |
| **Itérations lissage** | Passes de lissage Laplacien sur les bordures (réduit l'escalier) |
| **Force lissage** | Force du Laplacien (0 = aucun, 1 = max) |
| **Ajouter flancs + fond** | Ajoute un GeoNodes qui extrude les bords vers Floor Z et bouche le fond |
| **Floor Z** | Hauteur Z du fond (world) |

## Méthode

Pour chaque vertex d'une grille fine (par défaut 1600 × 800) couvrant l'emprise XY de la cible :
1. **Ray-cast vertical** depuis l'origine Z vers le bas, contre un BVH de la cible
2. Si le rayon **touche** la cible → vertex snappé au point d'impact (XY ajusté + Z = hauteur de la cible à cet endroit)
3. Si **pas de touche** → vertex laissé à Z=0
4. **Filtrage** : faces dont tous les verts sont restés à Z=0 supprimées (= hors silhouette)
5. **Anti-escalier** (optionnel) : snap des verts de bord sur les arêtes de bordure réelles de la cible + lissage Laplacien

Le `Add Sides` GeoNodes en post extrude les bords vers un Z de fond et bouche le contour final pour obtenir un volume fermé prêt pour l'impression 3D.

## Licence

MIT
