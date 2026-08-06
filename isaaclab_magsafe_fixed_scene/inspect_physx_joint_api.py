from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
from pxr import PhysxSchema
print("PhysxJointAPI_present", hasattr(PhysxSchema, "PhysxJointAPI"), flush=True)
if hasattr(PhysxSchema, "PhysxJointAPI"):
    print(
        [name for name in dir(PhysxSchema.PhysxJointAPI)
         if "Projection" in name or "Tolerance" in name],
        flush=True,
    )
app.close()
