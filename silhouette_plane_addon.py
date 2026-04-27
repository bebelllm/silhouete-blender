bl_info = {
    "name": "Plan Silhouette",
    "author": "Mika",
    "version": (1, 3, 0),
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


def extract_top_surface(target, name, normal_z_threshold=-0.5, use_modifiers=True):
    """Extrait directement les faces de target dont la normale.z > seuil.
    Si use_modifiers=True, utilise le mesh ÉVALUÉ (avec modificateurs appliqués)."""
    # Récupère le mesh à utiliser
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

    # bmesh : supprime les faces non voulues + verts orphelins
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.transform(target.matrix_world)
    bm.normal_update()
    bm.faces.ensure_lookup_table()

    to_remove = [f for f in bm.faces if f.normal.z <= normal_z_threshold]
    bmesh.ops.delete(bm, geom=to_remove, context='FACES')
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')

    n_kept = len(bm.faces)
    bm.to_mesh(me)
    bm.free()
    # Reset matrix : la mesh est déjà en world coords après bm.transform()
    import mathutils
    obj.matrix_world = mathutils.Matrix.Identity(4)

    return obj, n_kept


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
                # Extraction directe des faces du dessus de la cible (fidélité max, zéro escalier)
                obj, n_faces = extract_top_surface(
                    target=s.target,
                    name=s.output_name,
                    normal_z_threshold=s.normal_z_threshold,
                    use_modifiers=s.use_modifiers,
                )
                msg_extra = f"{n_faces} faces extraites"
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
            # En mode EXTRACT, le mesh peut être très lourd (1M+ faces) et le GeoNodes
            # Mesh→Curve+Fill plante. Si > 300k faces, on prévient et on skip.
            if s.mode == 'EXTRACT' and len(obj.data.polygons) > 300_000:
                self.report({'WARNING'},
                    f"Mesh trop dense ({len(obj.data.polygons)}f) — flancs+fond skipés "
                    f"(risque crash). Décime d'abord puis ferme manuellement.")
            else:
                try:
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
            ('RAYCAST', "Ray-cast grille", "Grille subdivisée + ray-cast (résolution réglable, peut générer un escalier sur les bords)"),
            ('EXTRACT', "Extraire surface source", "Copie directe des faces du dessus de la cible (fidélité parfaite, pas d'escalier, mais hérite de la topologie source)"),
        ],
        default='RAYCAST',
        description="Méthode de génération du plan",
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

        if s.mode == 'EXTRACT':
            box = layout.box()
            box.label(text="Extraction directe")
            box.prop(s, "normal_z_threshold")
            box.prop(s, "use_modifiers")
            box.label(text="Aucune approximation, fidélité parfaite", icon='CHECKMARK')
            box.label(text="Topologie héritée de la source", icon='ERROR')
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
