import bpy 

obj = bpy.context.object
obj.location[0] = 0.0
obj.keyframe_insert(data_path="location", frame=0, index=0)
obj.location[0] = 10.0
obj.keyframe_insert(data_path="location", frame=240, index=0)

bpy.ops.export_scene.gltf(
    filepath='/home/smirnova/dev/blender/automation/ex2', 
    check_existing=False, 
    export_format='GLB', 
    
    export_cameras=True
    )
    