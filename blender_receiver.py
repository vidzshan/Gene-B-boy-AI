import bpy
import socket
import struct
import mathutils
import os

# --- SETTINGS ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5000
JOINT_COUNT = 25
SCALE_FACTOR = 1.0 

# Path to your refinement GLB (Absolute Path to UltraShape Source)
GLB_PATH = r"C:\Users\kos04\OneDrive\Desktop\vidz\GCN\UltraShape-1.0\outputs\refine_demo\charachter_refined.glb"

class UpdReceiverModal(bpy.types.Operator):
    """Runs a modal timer to listen for UDP packets"""
    bl_idname = "wm.udp_receiver_modal"
    bl_label = "Sasaki UDP Receiver"
    
    _timer = None
    _sock = None
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            try:
                data, addr = self._sock.recvfrom(1024) 
                if len(data) >= 300:
                    count = len(data) // 4
                    floats = struct.unpack(f'{count}f', data)
                    
                    # Update Blender Debug Spheres
                    for i in range(JOINT_COUNT):
                        if i*3 + 2 < len(floats):
                            # HPI-GCN output is typically (X, Y, Z) normalized [-1, 1]
                            # Blender is Z-Up. We usually map:
                            # Python X -> Blender X
                            # Python Y -> Blender Z (Height)
                            # Python Z -> Blender Y (Depth)
                            x = floats[i*3] * SCALE_FACTOR
                            y = floats[i*3+1] * SCALE_FACTOR # Height
                            z = floats[i*3+2] * SCALE_FACTOR # Depth
                            
                            # Update Empty position
                            obj_name = f"J{i}"
                            if obj_name in bpy.data.objects:
                                obj = bpy.data.objects[obj_name]
                                # Correct Mapping for Blender Viewport
                                obj.location = (x, z, y) 
                                
            except BlockingIOError:
                pass 
            except Exception as e:
                print(f"UDP Error: {e}")
                
        elif event.type == 'ESC':
            return self.cancel(context)
            
        return {'PASS_THROUGH'}

    def execute(self, context):
        # 0. Load the UltraShape Character if not present
        if not any("charachter" in o.name for o in bpy.data.objects):
            print(f"📂 Loading UltraShape: {GLB_PATH}")
            if os.path.exists(GLB_PATH):
                try:
                    bpy.ops.import_scene.gltf(filepath=GLB_PATH)
                    print("✅ UltraShape Loaded.")
                except RuntimeError as e:
                    print(f"⚠️ GLB Import Failed (Likely Corrupted): {e}")
                    print("➡️ Proceeding to start UDP Receiver anyway (Debug Spheres only).")
            else:
                print(f"⚠️ GLB Not Found: {GLB_PATH}")

        # 1. Setup Socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((UDP_IP, UDP_PORT))
        self._sock.setblocking(False)
        
        # 2. Setup Debug Skeleton Handles
        collection = bpy.data.collections.get("Sasaki_Rig")
        if not collection:
            collection = bpy.data.collections.new("Sasaki_Rig")
            bpy.context.scene.collection.children.link(collection)
            
        for i in range(JOINT_COUNT):
            name = f"J{i}"
            if name not in bpy.data.objects:
                bpy.ops.object.empty_add(type='SPHERE', radius=0.03)
                obj = bpy.context.active_object
                obj.name = name
                for coll in obj.users_collection:
                    coll.objects.unlink(obj)
                collection.objects.link(obj)
                
        # 3. Start Timer
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.016, window=context.window)
        wm.modal_handler_add(self)
        print(f"✅ Listening on {UDP_IP}:{UDP_PORT}")
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        self._sock.close()
        print("🛑 UDP Receiver Stopped")
        return {'CANCELLED'}

def register():
    bpy.utils.register_class(UpdReceiverModal)

def unregister():
    bpy.utils.unregister_class(UpdReceiverModal)

if __name__ == "__main__":
    register()
    bpy.ops.wm.udp_receiver_modal()
