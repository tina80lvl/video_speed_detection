#Import python
import bpy
from bpy import context, data, ops


mat = bpy.data.materials.new(name="New_Mat")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
texImage.image = bpy.data.images.load("/Users/tina/dev/simicon/plates/Y999YY25.jpg")
mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])

ob = context.view_layer.objects.active

# Assign it to object
if ob.data.materials:
    ob.data.materials[0] = mat
else:
    ob.data.materials.append(mat)
