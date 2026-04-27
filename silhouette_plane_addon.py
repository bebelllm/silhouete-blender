bl_info = {
    "name": "Plan Silhouette",
    "author": "Mika",
    "version": (1, 8, 1),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Silhouette",
    "description": "Crée un plan dont chaque vert est snappé sur la surface d'un objet cible (silhouette + relief Z) via ray-cast. Pratique pour extraire un bas-relief ou une heightmap topologique.",
    "category": "Mesh",
}

import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy.props import IntProperty, FloatProperty, StringProperty, PointerProperty, BoolProperty, EnumProperty


def _bbox_xy(obj):
    """Retourne (xmin, xmax, ymin, ymax) en world coords."""
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    return min(xs), max(xs), min(ys), max(ys)


def _snap_boundary_to_target_outline(plane_bm, target, max_dist=0.01):
    """Snap les verts de bordure du plan sur les arêtes de bordure (Z=floor) de target.
    Élimine l'effet escalier en alignant les bords du plan sur les vrais contours courbes."""
    from mathutils.kdtree import KDTree

    # Extraire les arêtes de bordure de target à Z bas
    bm_t = bmesh.new()
    bm_t.from_mesh(target.data)
    bm_t.transform(target.matrix_world)
    bm_t.edges.ensure_lookup_table()

    # On échantillonne les arêtes (plusieurs points par arête pour précision)
    samples = []
    for e in bm_t.edges:
        if len(e.link_faces) == 1:  # boundary edge
            v0, v1 = e.verts[0].co, e.verts[1].co
            if v0.z < 0.005 and v1.z < 0.005:
                # 5 échantillons par arête
                for t in (0.0, 0.25, 0.5, 0.75, 1.0):
                    samples.append(v0.lerp(v1, t))
    bm_t.free()

    if not samples:
        return 0

    kd = KDTree(len(samples))
    for i, p in enumerate(samples):
        kd.insert(p, i)
    kd.balance()

    # Identifier les verts de bordure du plan (faces voisines == 1)
    plane_bm.edges.ensure_lookup_table()
    boundary_verts = set()
    for e in plane_bm.edges:
        if len(e.link_faces) == 1:
            boundary_verts.add(e.verts[0])
            boundary_verts.add(e.verts[1])

    snapped = 0
    for v in boundary_verts:
        query = Vector((v.co.x, v.co.y, 0.0))
        loc, idx, dist = kd.find(query)
        if dist < max_dist:
            v.co.x = loc.x
            v.co.y = loc.y
            snapped += 1
    return snapped


def _laplacian_smooth_boundary(plane_bm, iterations=3, factor=0.5):
    """Lissage Laplacien des verts de bordure : chaque vert tend vers la moyenne XY de ses voisins de bord.
    Z préservé. Réduit l'effet escalier sans déformer la courbe globale."""
    plane_bm.edges.ensure_lookup_table()
    boundary_verts = set()
    for e in plane_bm.edges:
        if len(e.link_faces) == 1:
            boundary_verts.add(e.verts[0])
            boundary_verts.add(e.verts[1])

    # Pour chaque vert de bord, récupérer ses voisins de bord (les autres extrémités d'arêtes de bord)
    neighbors = {}
    for v in boundary_verts:
        nbs = []
        for e in v.link_edges:
            if len(e.link_faces) == 1:
                other = e.other_vert(v)
                if other is not None and other in boundary_verts:
                    nbs.append(other)
        neighbors[v] = nbs

    for _ in range(iterations):
        new_pos = {}
        for v, nbs in neighbors.items():
            if len(nbs) >= 2:
                avg_x = sum(n.co.x for n in nbs) / len(nbs)
                avg_y = sum(n.co.y for n in nbs) / len(nbs)
                new_pos[v] = (
                    v.co.x * (1 - factor) + avg_x * factor,
                    v.co.y * (1 - factor) + avg_y * factor,
                )
        for v, (x, y) in new_pos.items():
            v.co.x = x
            v.co.y = y


def _filter_interior_via_blender_op(obj):
    """Utilise l'opérateur natif Blender select_interior_faces() pour détecter les faces enfouies.
    Retourne le nombre de faces supprimées."""
    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.selected_objects: o.select_set(False)
    obj.select_set(True)

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.select_interior_faces()
    # Compter les sélectionnés
    n = sum(1 for f in obj.data.polygons if f.select)
    bpy.ops.mesh.delete(type='FACE')
    bpy.ops.object.mode_set(mode='OBJECT')
    return n


def _filter_visibility_multi_direction(bm, n_directions=8, escape_distance=100.0):
    """Pour chaque face, lance n_directions rayons depuis center+epsilon*normal vers diverses directions
    distribuées dans l'hémisphère. Si AUCUN rayon ne s'échappe → face complètement enfermée."""
    import math, random
    bvh = BVHTree.FromBMesh(bm)
    bm.faces.ensure_lookup_table()

    # Directions distribuées sur la demi-sphère
    base_dirs = []
    for k in range(n_directions):
        # Fibonacci sphere demi-haut
        phi = math.acos(1 - (k + 0.5) / n_directions)
        theta = math.pi * (1 + 5**0.5) * k
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        z = math.cos(phi)
        base_dirs.append(Vector((x, y, z)))

    eps = 0.0001
    interior = []
    for f in bm.faces:
        c = f.calc_center_median()
        n = f.normal
        if n.length < 1e-6:
            continue
        origin = c + n * eps
        # Aligner les directions sur l'hémisphère de la normale (axe Z local = n)
        # Trouver un axe perpendiculaire
        if abs(n.z) < 0.9:
            tx = n.cross(Vector((0,0,1))).normalized()
        else:
            tx = n.cross(Vector((1,0,0))).normalized()
        ty = n.cross(tx).normalized()

        any_escape = False
        for d in base_dirs:
            world_dir = (tx * d.x + ty * d.y + n * d.z).normalized()
            loc, _, idx, _ = bvh.ray_cast(origin, world_dir, escape_distance)
            if loc is None or idx == f.index:
                any_escape = True
                break
        if not any_escape:
            interior.append(f)

    if interior:
        bmesh.ops.delete(bm, geom=interior, context='FACES')
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
    return len(interior)


def _filter_topmost_faces(bm, ray_height=10.0):
    """Pour chaque face, lance un rayon vertical descendant depuis ray_height au-dessus de son centre.
    Garde la face SEULEMENT si elle est la première touchée par ce rayon (= visible du dessus).
    Élimine toutes les couches enfouies sous d'autres surfaces."""
    bvh = BVHTree.FromBMesh(bm)
    bm.faces.ensure_lookup_table()

    direction = Vector((0, 0, -1))
    not_topmost = []
    for f in bm.faces:
        c = f.calc_center_median()
        origin = Vector((c.x, c.y, ray_height))
        loc, hn, hit_idx, dist = bvh.ray_cast(origin, direction, ray_height + 100.0)
        # Si la première face touchée n'est pas celle-ci → enfouie
        if hit_idx is not None and hit_idx != f.index:
            not_topmost.append(f)

    if not_topmost:
        bmesh.ops.delete(bm, geom=not_topmost, context='FACES')
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
    return len(not_topmost)


def _filter_exterior_faces_by_raycast(bm, escape_distance=100.0):
    """Pour chaque face, ray-cast depuis center+epsilon*normal dans la direction de la normale,
    avec une très grande distance (= test infini en pratique).
    Si le rayon s'échappe → face extérieure (gardée). S'il tape n'importe quelle autre face → enfouie."""
    bvh = BVHTree.FromBMesh(bm)
    bm.faces.ensure_lookup_table()

    eps = 0.0001
    interior = []
    for f in bm.faces:
        center = f.calc_center_median()
        n = f.normal
        if n.length < 1e-6:
            continue
        origin = center + n * eps
        loc, hit_n, idx, dist = bvh.ray_cast(origin, n, escape_distance)
        if loc is not None and idx != f.index:
            interior.append(f)

    if interior:
        bmesh.ops.delete(bm, geom=interior, context='FACES')
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
    return len(interior)


def bake_clean_remesh(target, name, voxel_size=0.003, merge_threshold=0.0001,
                       fix_poles=True, preserve_volume=True,
                       smooth_shading=False, close_bottom=True, floor_z=0.0):
    """Pipeline 'mesh propre fermé' :
    1. Duplique le target avec modificateurs appliqués (mesh évalué)
    2. Merge by Distance pour éliminer les doublons exacts
    3. (optionnel) Ferme le fond : extrude boundaries vers floor_z + triangle_fill
    4. Voxel Remesh → shell extérieur manifold sans intérieur
    Le close_bottom est essentiel si le mesh source a le dessous ouvert,
    sinon le voxel remesh reproduit le shell ouvert."""
    deps = bpy.context.evaluated_depsgraph_get()
    ev = target.evaluated_get(deps)
    me = bpy.data.meshes.new_from_object(ev, depsgraph=deps)

    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    obj.matrix_world = target.matrix_world.copy()

    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.selected_objects: o.select_set(False)
    obj.select_set(True)

    # Merge by Distance
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=merge_threshold)
    bpy.ops.object.mode_set(mode='OBJECT')

    n_after_merge = len(obj.data.vertices)

    # Fermer le fond AVANT remesh (sinon remesh garde l'ouverture)
    n_filled = 0
    if close_bottom:
        n_filled = add_sides_bmesh(obj, floor_z=floor_z)

    # Voxel Remesh
    obj.data.remesh_voxel_size = voxel_size
    obj.data.use_remesh_fix_poles = fix_poles
    obj.data.use_remesh_preserve_volume = preserve_volume
    bpy.ops.object.voxel_remesh()

    if smooth_shading:
        for f in obj.data.polygons:
            f.use_smooth = True

    return obj, n_after_merge, n_filled


def extract_top_surface(target, name, normal_z_threshold=-0.5, use_modifiers=True,
                        exterior_only=False, escape_distance=0.005,
                        topmost_only=False, ray_height=10.0):
    """Extrait directement les faces de target dont la normale.z > seuil.
    Si use_modifiers=True, utilise le mesh ÉVALUÉ (avec modificateurs appliqués).
    Si exterior_only=True, supprime ensuite les faces enfouies (test ray-cast)."""
    if use_modifiers:
        deps = bpy.context.evaluated_depsgraph_get()
        ev = target.evaluated_get(deps)
        me = bpy.data.meshes.new_from_object(ev, depsgraph=deps)
    else:
        me = target.data.copy()

    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.transform(target.matrix_world)
    bm.normal_update()
    bm.faces.ensure_lookup_table()

    # Filtre 1 : seuil normale Z
    to_remove = [f for f in bm.faces if f.normal.z <= normal_z_threshold]
    bmesh.ops.delete(bm, geom=to_remove, context='FACES')
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')

    # Filtre 2 (optionnel) : extérieur seulement via ray-cast
    n_interior_removed = 0
    if exterior_only:
        n_interior_removed = _filter_exterior_faces_by_raycast(bm, escape_distance=escape_distance)

    # Filtre 3 (optionnel) : garder uniquement la peau topmost (la plus haute à chaque XY)
    if topmost_only:
        n_buried = _filter_topmost_faces(bm, ray_height=ray_height)
        n_interior_removed += n_buried

    n_kept = len(bm.faces)
    bm.to_mesh(me)
    bm.free()
    import mathutils
    obj.matrix_world = mathutils.Matrix.Identity(4)

    return obj, n_kept, n_interior_removed


def _raycast_grid(bvh, axis, direction_sign, plane_origin, u_min, u_max, v_min, v_max, res_u, res_v, max_dist):
    """Génère une grille perpendiculaire à `axis` ('X','Y','Z') et ray-caste dans le sens direction_sign.
    Retourne un bmesh contenant uniquement les verts/faces qui ont hit la cible.
    `axis` = direction du rayon, `plane_origin` = position du plan sur cet axe.
    `u_min..v_max` = bornes 2D dans les axes perpendiculaires."""
    bm = bmesh.new()
    grid = []
    direction = Vector((0, 0, 0))
    if axis == 'X':
        direction.x = direction_sign
        u_axis_idx, v_axis_idx, fixed_idx = 1, 2, 0  # u=Y, v=Z
    elif axis == 'Y':
        direction.y = direction_sign
        u_axis_idx, v_axis_idx, fixed_idx = 0, 2, 1  # u=X, v=Z
    else:  # Z
        direction.z = direction_sign
        u_axis_idx, v_axis_idx, fixed_idx = 0, 1, 2  # u=X, v=Y

    for j in range(res_v):
        row = []
        v = v_min + (v_max - v_min) * j / (res_v - 1)
        for i in range(res_u):
            u = u_min + (u_max - u_min) * i / (res_u - 1)
            co = [0.0, 0.0, 0.0]
            co[u_axis_idx] = u
            co[v_axis_idx] = v
            co[fixed_idx] = plane_origin
            row.append(bm.verts.new(co))
        grid.append(row)
    bm.verts.ensure_lookup_table()
    for j in range(res_v - 1):
        for i in range(res_u - 1):
            bm.faces.new([grid[j][i], grid[j][i+1], grid[j+1][i+1], grid[j+1][i]])

    # Ray-cast chaque vert
    for v in bm.verts:
        loc, n, idx, dist = bvh.ray_cast(v.co, direction, max_dist)
        if loc is not None:
            v.co = loc.copy()

    # Supprimer les faces dont aucun vert n'a hit (verts qui sont restés sur le plan d'origine)
    bm.faces.ensure_lookup_table()
    to_remove = [f for f in bm.faces if all(abs(vv.co[fixed_idx] - plane_origin) < 0.0001 for vv in f.verts)]
    bmesh.ops.delete(bm, geom=to_remove, context='FACES')
    loose = [vv for vv in bm.verts if not vv.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')

    return bm


def build_multidir_silhouette(target, bounds_obj, res_top, res_side, max_dist, name,
                               cast_top=True, cast_bottom=False,
                               cast_xneg=True, cast_xpos=True,
                               cast_yneg=True, cast_ypos=True,
                               use_modifiers=True, merge_threshold=0.001):
    """Lance plusieurs ray-casts (top + 4 côtés + bottom optionnel) et combine les hits dans un seul mesh.
    Couvre les flancs verticaux que la simple grille du dessus ne capture pas."""
    # Mesh évalué
    if use_modifiers:
        deps = bpy.context.evaluated_depsgraph_get()
        ev = target.evaluated_get(deps)
        src_me = bpy.data.meshes.new_from_object(ev, depsgraph=deps)
    else:
        src_me = target.data.copy()

    # BVH en world coords
    bm_t = bmesh.new()
    bm_t.from_mesh(src_me)
    bm_t.transform(target.matrix_world)
    bvh = BVHTree.FromBMesh(bm_t)

    # Bbox world (depuis bounds_obj ou bbox cible)
    src = bounds_obj if bounds_obj is not None else target
    corners = [src.matrix_world @ Vector(c) for c in src.bound_box]
    if src is target and bounds_obj is None:
        # bbox réelle de la mesh évaluée
        if bm_t.verts:
            xs = [v.co.x for v in bm_t.verts]
            ys = [v.co.y for v in bm_t.verts]
            zs = [v.co.z for v in bm_t.verts]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            zmin, zmax = min(zs), max(zs)
        else:
            xmin = xmax = ymin = ymax = zmin = zmax = 0
    else:
        xs = [c.x for c in corners]; ys = [c.y for c in corners]; zs = [c.z for c in corners]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        zmin, zmax = min(zs), max(zs)
    bm_t.free()

    # Petite marge pour les origines de plans
    margin = max((xmax-xmin), (ymax-ymin), (zmax-zmin)) * 0.1 + 0.01

    bm_final = bmesh.new()

    def merge_bm(src_bm):
        # Ajouter les verts/faces de src_bm dans bm_final
        vmap = {}
        for v in src_bm.verts:
            vmap[v] = bm_final.verts.new(v.co.copy())
        for f in src_bm.faces:
            try:
                bm_final.faces.new([vmap[v] for v in f.verts])
            except ValueError:
                pass
        src_bm.free()

    # TOP : plan à Z=zmax+margin, rayons vers le bas
    if cast_top:
        b = _raycast_grid(bvh, 'Z', -1, zmax + margin,
                          xmin, xmax, ymin, ymax, res_top, int(res_top * (ymax-ymin) / (xmax-xmin)) if (xmax > xmin) else res_top,
                          max_dist)
        merge_bm(b)

    # BOTTOM
    if cast_bottom:
        b = _raycast_grid(bvh, 'Z', 1, zmin - margin,
                          xmin, xmax, ymin, ymax, res_top, int(res_top * (ymax-ymin) / (xmax-xmin)) if (xmax > xmin) else res_top,
                          max_dist)
        merge_bm(b)

    # SIDES : plans verticaux
    side_res_v = res_side  # vertical resolution = side height
    if cast_xneg:
        b = _raycast_grid(bvh, 'X', 1, xmin - margin,
                          ymin, ymax, zmin, zmax, res_side, side_res_v, max_dist)
        merge_bm(b)
    if cast_xpos:
        b = _raycast_grid(bvh, 'X', -1, xmax + margin,
                          ymin, ymax, zmin, zmax, res_side, side_res_v, max_dist)
        merge_bm(b)
    if cast_yneg:
        b = _raycast_grid(bvh, 'Y', 1, ymin - margin,
                          xmin, xmax, zmin, zmax, res_side, side_res_v, max_dist)
        merge_bm(b)
    if cast_ypos:
        b = _raycast_grid(bvh, 'Y', -1, ymax + margin,
                          xmin, xmax, zmin, zmax, res_side, side_res_v, max_dist)
        merge_bm(b)

    # Souder les doublons à la jonction des 5/6 plans
    bmesh.ops.remove_doubles(bm_final, verts=list(bm_final.verts), dist=merge_threshold)

    # Output object
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    me = bpy.data.meshes.new(name + "_mesh")
    bm_final.to_mesh(me)
    n_verts = len(bm_final.verts); n_faces = len(bm_final.faces)
    bm_final.free()
    bpy.data.meshes.remove(src_me)

    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)

    return obj, n_verts, n_faces


def build_silhouette_plane(target, bounds_obj, res_x, res_y, cast_height, cast_distance, name,
                           smooth_borders=False, smooth_max_dist=0.01,
                           laplacian_iters=0, laplacian_factor=0.5):
    """Crée un objet plan dont chaque vert est snappé via ray-cast vers le bas sur la surface de `target`.
    Les verts qui n'ont pas hit sont supprimés (avec leurs faces incidentes).
    Si smooth_borders, snappe les verts de bordure sur les vrais contours de la cible (anti-escalier).
    """
    # Bornes XY depuis bounds_obj (ou bbox target si None)
    src = bounds_obj if bounds_obj is not None else target
    xmin, xmax, ymin, ymax = _bbox_xy(src)

    # Grille
    bm = bmesh.new()
    grid = []
    for j in range(res_y):
        row = []
        y = ymin + (ymax - ymin) * j / (res_y - 1)
        for i in range(res_x):
            x = xmin + (xmax - xmin) * i / (res_x - 1)
            v = bm.verts.new((x, y, 0.0))
            row.append(v)
        grid.append(row)
    bm.verts.ensure_lookup_table()
    for j in range(res_y - 1):
        for i in range(res_x - 1):
            bm.faces.new([grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]])

    # BVH cible (en world)
    bm_t = bmesh.new()
    bm_t.from_mesh(target.data)
    bm_t.transform(target.matrix_world)
    bvh = BVHTree.FromBMesh(bm_t)
    bm_t.free()

    # Ray-cast vers le bas et snap
    direction = Vector((0, 0, -1))
    snapped = 0
    for v in bm.verts:
        origin = Vector((v.co.x, v.co.y, cast_height))
        loc, n, idx, dist = bvh.ray_cast(origin, direction, cast_distance)
        if loc is not None:
            v.co = Vector((loc.x, loc.y, loc.z))
            snapped += 1

    # Filtre faces hors silhouette : faces dont tous les verts sont à Z=0 (= aucun hit)
    bm.faces.ensure_lookup_table()
    to_remove = [f for f in bm.faces if all(v.co.z < 0.0001 for v in f.verts)]
    bmesh.ops.delete(bm, geom=to_remove, context='FACES')
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context='VERTS')

    # Anti-escalier : snap des bordures sur les vrais contours de la cible
    border_snapped = 0
    if smooth_borders:
        border_snapped = _snap_boundary_to_target_outline(bm, target, max_dist=smooth_max_dist)

    # Lissage Laplacien itératif (lissage XY uniquement, Z préservé)
    if laplacian_iters > 0:
        _laplacian_smooth_boundary(bm, iterations=laplacian_iters, factor=laplacian_factor)

    # Crée objet
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    me = bpy.data.meshes.new(name + "_mesh")
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)

    return obj, snapped, border_snapped


def add_sides_bmesh(obj, floor_z=0.0):
    """Ferme le mesh en pur bmesh : extrude boundary edges vers floor_z + remplit le fond.
    Beaucoup plus rapide que la version GeoNodes (pas de Mesh→Curve+Fill Curve)."""
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    # 1. Boundary edges (face_count == 1)
    boundary_edges = [e for e in bm.edges if len(e.link_faces) == 1]
    if not boundary_edges:
        bm.free()
        return 0

    # 2. Extrude ces edges. Le résultat duplique les verts ; on déplace les nouveaux à floor_z
    ret = bmesh.ops.extrude_edge_only(bm, edges=boundary_edges)
    new_verts = [g for g in ret['geom'] if isinstance(g, bmesh.types.BMVert)]
    for v in new_verts:
        v.co.z = floor_z

    # 3. Récup edges au floor_z toujours boundary → fill
    bm.edges.ensure_lookup_table()
    floor_edges = [e for e in bm.edges
                   if abs(e.verts[0].co.z - floor_z) < 0.0001
                   and abs(e.verts[1].co.z - floor_z) < 0.0001
                   and len(e.link_faces) == 1]

    n_filled = 0
    if floor_edges:
        try:
            res = bmesh.ops.triangle_fill(bm, edges=floor_edges, use_beauty=True,
                                           normal=Vector((0, 0, -1)))
            n_filled = sum(1 for g in res.get('geom', []) if isinstance(g, bmesh.types.BMFace))
        except Exception:
            pass

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    return n_filled


def add_sides_geonodes(obj, floor_z=0.0):
    """Ajoute un Geometry Nodes modifier qui extrude les bords vers Z=floor_z + ferme le fond."""
    ng_name = "SilhouettePlaneSides"
    if ng_name not in bpy.data.node_groups:
        ng = bpy.data.node_groups.new(ng_name, "GeometryNodeTree")
        ng.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        ng.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
        ng.interface.new_socket(name="Floor Z", in_out='INPUT', socket_type='NodeSocketFloat')

        nodes = ng.nodes
        links = ng.links

        n_in = nodes.new("NodeGroupInput"); n_in.location = (-1000, 0)
        n_out = nodes.new("NodeGroupOutput"); n_out.location = (1000, 0)

        en = nodes.new("GeometryNodeInputMeshEdgeNeighbors"); en.location = (-800, -300)
        cmp = nodes.new("FunctionNodeCompare"); cmp.location = (-600, -300)
        cmp.data_type = 'INT'; cmp.operation = 'EQUAL'
        cmp.inputs[3].default_value = 1
        links.new(en.outputs["Face Count"], cmp.inputs[2])

        ext = nodes.new("GeometryNodeExtrudeMesh"); ext.location = (-300, 0)
        ext.mode = 'EDGES'
        ext.inputs["Offset Scale"].default_value = 0.0
        links.new(n_in.outputs["Geometry"], ext.inputs["Mesh"])
        links.new(cmp.outputs[0], ext.inputs["Selection"])

        sp = nodes.new("GeometryNodeSetPosition"); sp.location = (0, 0)
        links.new(ext.outputs["Mesh"], sp.inputs["Geometry"])
        links.new(ext.outputs["Top"], sp.inputs["Selection"])

        pos = nodes.new("GeometryNodeInputPosition"); pos.location = (-300, -300)
        sep = nodes.new("ShaderNodeSeparateXYZ"); sep.location = (-100, -300)
        links.new(pos.outputs["Position"], sep.inputs[0])

        cb = nodes.new("ShaderNodeCombineXYZ"); cb.location = (100, -300)
        links.new(sep.outputs["X"], cb.inputs[0])
        links.new(sep.outputs["Y"], cb.inputs[1])
        links.new(n_in.outputs["Floor Z"], cb.inputs[2])
        links.new(cb.outputs[0], sp.inputs["Position"])

        # Mesh→Curve sur arêtes du fond → Fill Curve
        ev = nodes.new("GeometryNodeInputMeshEdgeVertices"); ev.location = (200, -550)
        sepA = nodes.new("ShaderNodeSeparateXYZ"); sepA.location = (350, -550)
        sepB = nodes.new("ShaderNodeSeparateXYZ"); sepB.location = (350, -700)
        links.new(ev.outputs["Position 1"], sepA.inputs[0])
        links.new(ev.outputs["Position 2"], sepB.inputs[0])

        add_eps = nodes.new("ShaderNodeMath"); add_eps.location = (400, -650)
        add_eps.operation = 'ADD'; add_eps.inputs[1].default_value = 0.001
        links.new(n_in.outputs["Floor Z"], add_eps.inputs[0])

        cmpA = nodes.new("FunctionNodeCompare"); cmpA.location = (550, -550)
        cmpA.data_type = 'FLOAT'; cmpA.operation = 'LESS_THAN'
        links.new(sepA.outputs["Z"], cmpA.inputs[0])
        links.new(add_eps.outputs[0], cmpA.inputs[1])
        cmpB = nodes.new("FunctionNodeCompare"); cmpB.location = (550, -700)
        cmpB.data_type = 'FLOAT'; cmpB.operation = 'LESS_THAN'
        links.new(sepB.outputs["Z"], cmpB.inputs[0])
        links.new(add_eps.outputs[0], cmpB.inputs[1])

        bool_and = nodes.new("FunctionNodeBooleanMath"); bool_and.location = (700, -600)
        bool_and.operation = 'AND'
        links.new(cmpA.outputs[0], bool_and.inputs[0])
        links.new(cmpB.outputs[0], bool_and.inputs[1])

        m2c = nodes.new("GeometryNodeMeshToCurve"); m2c.location = (300, 200)
        links.new(sp.outputs["Geometry"], m2c.inputs["Mesh"])
        links.new(bool_and.outputs[0], m2c.inputs["Selection"])

        fill = nodes.new("GeometryNodeFillCurve"); fill.location = (500, 200)
        links.new(m2c.outputs["Curve"], fill.inputs["Curve"])

        join = nodes.new("GeometryNodeJoinGeometry"); join.location = (750, 0)
        links.new(sp.outputs["Geometry"], join.inputs["Geometry"])
        links.new(fill.outputs["Mesh"], join.inputs["Geometry"])

        links.new(join.outputs["Geometry"], n_out.inputs["Geometry"])
    else:
        ng = bpy.data.node_groups[ng_name]

    mod = obj.modifiers.new(name="SilhouetteSides", type='NODES')
    mod.node_group = ng
    for item in ng.interface.items_tree:
        if hasattr(item, 'in_out') and item.in_out == 'INPUT' and item.socket_type == 'NodeSocketFloat':
            mod[item.identifier] = floor_z


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class SILH_OT_create_plane(bpy.types.Operator):
    bl_idname = "object.silhouette_plane_create"
    bl_label = "Créer Plan Silhouette"
    bl_description = "Génère un plan ray-casté sur la cible (silhouette + relief Z)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.silhouette_settings

        if s.target is None:
            self.report({'ERROR'}, "Pas de cible définie")
            return {'CANCELLED'}
        if s.target.type != 'MESH':
            self.report({'ERROR'}, "La cible doit être un mesh")
            return {'CANCELLED'}

        try:
            if s.mode == 'EXTRACT':
                obj, n_faces, n_interior = extract_top_surface(
                    target=s.target,
                    name=s.output_name,
                    normal_z_threshold=s.normal_z_threshold,
                    use_modifiers=s.use_modifiers,
                    exterior_only=s.exterior_only,
                    escape_distance=s.escape_distance,
                    topmost_only=s.topmost_only,
                    ray_height=s.cast_height,
                )
                msg_extra = f"{n_faces} faces extraites"
                if s.exterior_only or s.topmost_only:
                    msg_extra += f", {n_interior} enfouies supprimées"
            elif s.mode == 'BAKE_REMESH':
                obj, n_merged, n_filled = bake_clean_remesh(
                    target=s.target,
                    name=s.output_name,
                    voxel_size=s.voxel_size,
                    merge_threshold=s.merge_threshold,
                    fix_poles=s.remesh_fix_poles,
                    preserve_volume=s.remesh_preserve_volume,
                    smooth_shading=s.smooth_shading,
                    close_bottom=s.bake_close_bottom,
                    floor_z=s.floor_z,
                )
                msg_extra = f"merge → {n_merged}v, fond fermé ({n_filled}f), remesh → {len(obj.data.vertices)}v {len(obj.data.polygons)}f"
            elif s.mode == 'MULTIDIR':
                obj, n_v, n_f = build_multidir_silhouette(
                    target=s.target,
                    bounds_obj=s.bounds_object,
                    res_top=s.res_x,
                    res_side=s.res_side,
                    max_dist=s.cast_distance,
                    name=s.output_name,
                    cast_top=s.cast_top, cast_bottom=s.cast_bottom,
                    cast_xneg=s.cast_xneg, cast_xpos=s.cast_xpos,
                    cast_yneg=s.cast_yneg, cast_ypos=s.cast_ypos,
                    use_modifiers=s.use_modifiers,
                    merge_threshold=s.merge_threshold,
                )
                msg_extra = f"{n_v}v {n_f}f multi-dir"
            else:
                # Mode RAYCAST : grille + ray-cast
                obj, hits, border_snapped = build_silhouette_plane(
                    target=s.target,
                    bounds_obj=s.bounds_object,
                    res_x=s.res_x,
                    res_y=s.res_y,
                    cast_height=s.cast_height,
                    cast_distance=s.cast_distance,
                    name=s.output_name,
                    smooth_borders=s.smooth_borders,
                    smooth_max_dist=s.smooth_max_dist,
                    laplacian_iters=s.laplacian_iters,
                    laplacian_factor=s.laplacian_factor,
                )
                msg_extra = f"{hits} hits"
                if s.smooth_borders:
                    msg_extra += f", {border_snapped} bordures snappées"
        except Exception as e:
            self.report({'ERROR'}, f"Erreur génération: {e}")
            return {'CANCELLED'}

        if s.add_sides:
            try:
                if s.mode == 'EXTRACT':
                    # bmesh direct (rapide, gère les meshes 1M+ faces sans crash)
                    n_filled = add_sides_bmesh(obj, floor_z=s.floor_z)
                    msg_extra += f", fond fermé ({n_filled} faces)"
                else:
                    # GeoNodes non-destructif (modificateur ajustable)
                    add_sides_geonodes(obj, floor_z=s.floor_z)
            except Exception as e:
                self.report({'WARNING'}, f"Plan généré mais sides échoué: {e}")

        for o in bpy.context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj

        self.report({'INFO'}, f"{obj.name}: {len(obj.data.vertices)}v {len(obj.data.polygons)}f ({msg_extra})")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _mesh_object_poll(self, obj):
    return obj.type == 'MESH'


class SILH_settings(bpy.types.PropertyGroup):
    mode: EnumProperty(
        name="Mode",
        items=[
            ('RAYCAST', "Ray-cast top", "Grille du haut + ray-cast vertical (top seulement)"),
            ('MULTIDIR', "Ray-cast multi-direction", "Ray-cast depuis le haut + 4 côtés (top + flancs, capture la peau extérieure complète)"),
            ('EXTRACT', "Extraire surface source", "Copie directe des faces de la cible (fidélité parfaite, hérite topologie source, peut garder des faces enfouies)"),
            ('BAKE_REMESH', "Bake & Voxel Remesh", "Applique modificateurs + Merge by Distance + Voxel Remesh. Donne un shell extérieur manifold propre. Idéal après BASIFY MAKE MANIFOLD."),
        ],
        default='RAYCAST',
        description="Méthode de génération du plan",
    )
    voxel_size: FloatProperty(
        name="Voxel size", default=0.005, min=0.001, max=1.0, precision=4,
        description="Taille du voxel pour le remesh. Plus petit = plus précis mais plus lourd. 0.005 = 5mm (sûr). <0.003 risque de freezer Blender sur grands meshes (3m×1.5m).",
    )
    bake_close_bottom: BoolProperty(
        name="Fermer le fond avant remesh",
        default=True,
        description="Extrude les boundaries vers Floor Z et triangule, AVANT le voxel remesh. Indispensable si le mesh source a le dessous ouvert (sinon le remesh reproduit l'ouverture).",
    )
    remesh_fix_poles: BoolProperty(
        name="Fix Poles", default=True,
        description="Corrige les pôles dans la topo voxel remesh",
    )
    remesh_preserve_volume: BoolProperty(
        name="Preserve Volume", default=True,
        description="Conserve le volume original (compense le shrink du voxel grid)",
    )
    smooth_shading: BoolProperty(
        name="Smooth Shading", default=False,
        description="Active shade smooth sur les faces du résultat",
    )
    res_side: IntProperty(
        name="Résolution flancs", default=400, min=10, max=4000,
        description="Résolution des grilles latérales (multi-direction)",
    )
    cast_top: BoolProperty(name="Top", default=True, description="Ray-cast depuis le haut")
    cast_bottom: BoolProperty(name="Bottom", default=False, description="Ray-cast depuis le bas")
    cast_xneg: BoolProperty(name="-X", default=True, description="Ray-cast depuis -X")
    cast_xpos: BoolProperty(name="+X", default=True, description="Ray-cast depuis +X")
    cast_yneg: BoolProperty(name="-Y", default=True, description="Ray-cast depuis -Y")
    cast_ypos: BoolProperty(name="+Y", default=True, description="Ray-cast depuis +Y")
    merge_threshold: FloatProperty(
        name="Seuil fusion jonctions",
        default=0.001, min=0.00001, max=0.1, precision=5,
        description="Distance pour souder les verts aux jonctions entre les grilles top/côtés",
    )
    normal_z_threshold: FloatProperty(
        name="Seuil normal Z",
        default=-0.5, min=-1.0, max=1.0,
        description="Garde les faces dont la normale.z > seuil. -0.5 = tops + flancs (recommandé). 0.1 = tops seulement (sans flancs). -0.99 = tout sauf le dessous strict.",
    )
    use_modifiers: BoolProperty(
        name="Avec modificateurs",
        default=True,
        description="Utilise le mesh évalué (avec modificateurs appliqués) plutôt que le mesh source brut. Important si la cible a Solidify, Subsurf, BASIFY, etc.",
    )
    exterior_only: BoolProperty(
        name="Extérieur seulement",
        default=False,
        description="Supprime les faces enfouies (intérieures aux coquilles superposées) via ray-cast. Plus lent mais propre sur les meshes avec shells doublés.",
    )
    topmost_only: BoolProperty(
        name="Topmost seulement",
        default=True,
        description="Pour chaque face, ray-cast vertical depuis le haut. Garde uniquement les faces qui sont les premières touchées (= peau du dessus visible). Élimine TOUTES les couches enfouies.",
    )
    escape_distance: FloatProperty(
        name="Distance test",
        default=0.005, min=0.0001, max=1.0, precision=4,
        description="Distance max du ray-cast pour décider si une face est intérieure. Augmenter si shells très proches, diminuer si fines feuilles parallèles légitimes.",
    )
    target: PointerProperty(
        name="Cible",
        description="Objet mesh source",
        type=bpy.types.Object,
        poll=_mesh_object_poll,
    )
    bounds_object: PointerProperty(
        name="Limites XY",
        description="Objet définissant l'emprise XY (par défaut = bbox de la cible)",
        type=bpy.types.Object,
    )
    res_x: IntProperty(
        name="Résolution X", default=1600, min=10, max=10000,
        description="Subdivisions horizontales de la grille",
    )
    res_y: IntProperty(
        name="Résolution Y", default=800, min=10, max=10000,
        description="Subdivisions verticales de la grille",
    )
    cast_height: FloatProperty(
        name="Origine Z des rayons", default=1.0, min=-100.0, max=100.0,
        description="Hauteur Z d'où partent les rayons (doit être au-dessus de la cible)",
    )
    cast_distance: FloatProperty(
        name="Distance max", default=2.0, min=0.001, max=1000.0,
        description="Longueur max d'un rayon",
    )
    output_name: StringProperty(
        name="Nom sortie", default="plan_silhouettes",
        description="Nom de l'objet créé (écrasé si existe)",
    )
    smooth_borders: BoolProperty(
        name="Lisser les bordures",
        default=True,
        description="Snap les verts de bordure du plan sur les vrais contours courbes de la cible (élimine l'effet escalier)",
    )
    smooth_max_dist: FloatProperty(
        name="Distance snap max", default=0.01, min=0.0001, max=1.0, precision=4,
        description="Distance max pour qu'un vert de bord soit snappé sur le contour cible",
    )
    laplacian_iters: IntProperty(
        name="Itérations lissage", default=2, min=0, max=20,
        description="Nombre de passes de lissage Laplacien sur les bordures (réduit l'escalier)",
    )
    laplacian_factor: FloatProperty(
        name="Force lissage", default=0.5, min=0.0, max=1.0,
        description="Force d'attraction vers la moyenne des voisins (0=aucun, 1=plein)",
    )
    add_sides: BoolProperty(
        name="Ajouter flancs + fond",
        default=False,
        description="Ajoute un GeoNodes qui extrude les bords vers Floor Z et bouche le fond",
    )
    floor_z: FloatProperty(
        name="Floor Z", default=0.0, min=-100.0, max=100.0,
        description="Hauteur Z du fond (world)",
    )


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class SILH_PT_panel(bpy.types.Panel):
    bl_label = "Plan Silhouette"
    bl_idname = "SILH_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Silhouette"

    def draw(self, context):
        layout = self.layout
        s = context.scene.silhouette_settings

        layout.prop(s, "mode", expand=True)

        box = layout.box()
        box.label(text="Cible")
        box.prop(s, "target", text="")

        if s.mode == 'BAKE_REMESH':
            box = layout.box()
            box.label(text="Bake & Voxel Remesh")
            box.prop(s, "voxel_size")
            box.prop(s, "merge_threshold")
            box.prop(s, "bake_close_bottom")
            sub = box.row(); sub.enabled = s.bake_close_bottom
            sub.prop(s, "floor_z")
            box.prop(s, "remesh_fix_poles")
            box.prop(s, "remesh_preserve_volume")
            box.prop(s, "smooth_shading")
            box.label(text="Shell extérieur manifold garanti", icon='CHECKMARK')
            box.label(text="Voxel <0.002 risque crash sur gros mesh", icon='ERROR')
        elif s.mode == 'EXTRACT':
            box = layout.box()
            box.label(text="Extraction directe")
            box.prop(s, "normal_z_threshold")
            box.prop(s, "use_modifiers")
            box.prop(s, "topmost_only")
            box.prop(s, "exterior_only")
            sub = box.row()
            sub.enabled = s.exterior_only
            sub.prop(s, "escape_distance")
            box.label(text="Aucune approximation, fidélité parfaite", icon='CHECKMARK')
            if s.topmost_only:
                box.label(text="Topmost: élimine couches enfouies", icon='CHECKMARK')
        elif s.mode == 'MULTIDIR':
            box = layout.box()
            box.label(text="Cible & Limites")
            box.prop(s, "bounds_object", text="Limites XY")

            box = layout.box()
            box.label(text="Résolution")
            box.prop(s, "res_x", text="Top (haut/bas)")
            box.prop(s, "res_side")

            box = layout.box()
            box.label(text="Directions ray-cast")
            row = box.row(align=True); row.prop(s, "cast_top"); row.prop(s, "cast_bottom")
            row = box.row(align=True); row.prop(s, "cast_xneg"); row.prop(s, "cast_xpos")
            row = box.row(align=True); row.prop(s, "cast_yneg"); row.prop(s, "cast_ypos")

            box = layout.box()
            box.prop(s, "use_modifiers")
            box.prop(s, "cast_distance")
            box.prop(s, "merge_threshold")
            box.label(text="Capture peau extérieure (top + flancs)", icon='CHECKMARK')
            box.label(text="Pas de surfaces enfouies", icon='CHECKMARK')
        else:
            box = layout.box()
            box.label(text="Cible")
            box.prop(s, "bounds_object", text="Limites XY")

            box = layout.box()
            box.label(text="Résolution grille")
            row = box.row(align=True)
            row.prop(s, "res_x")
            row.prop(s, "res_y")
            box.label(text=f"Verts grille: {s.res_x * s.res_y:,}", icon='INFO')

            box = layout.box()
            box.label(text="Ray-cast")
            box.prop(s, "cast_height")
            box.prop(s, "cast_distance")

            box = layout.box()
            box.label(text="Lissage bordures (anti-escalier)")
            box.prop(s, "smooth_borders")
            sub = box.row()
            sub.enabled = s.smooth_borders
            sub.prop(s, "smooth_max_dist")
            box.prop(s, "laplacian_iters")
            sub2 = box.row()
            sub2.enabled = s.laplacian_iters > 0
            sub2.prop(s, "laplacian_factor")

        box = layout.box()
        box.label(text="Flancs (optionnel)")
        box.prop(s, "add_sides")
        sub = box.row()
        sub.enabled = s.add_sides
        sub.prop(s, "floor_z")

        layout.prop(s, "output_name")
        layout.operator("object.silhouette_plane_create", icon='MESH_PLANE')


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

classes = (SILH_settings, SILH_OT_create_plane, SILH_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.silhouette_settings = PointerProperty(type=SILH_settings)


def unregister():
    del bpy.types.Scene.silhouette_settings
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
