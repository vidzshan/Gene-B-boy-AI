import sounddevice as sd

print("Searching for Loopback/Stereo Mix devices...")
wasapi_host_id = -1
for i, api in enumerate(sd.query_hostapis()):
    if 'WASAPI' in api['name']:
        wasapi_host_id = i
        print(f"Found WASAPI Host: ID {i}")
        break

if wasapi_host_id != -1:
    default_output = sd.query_hostapis()[wasapi_host_id]['default_output_device']
    print(f"Default WASAPI Output Device ID: {default_output}")
    try:
        dev_info = sd.query_devices(default_output)
        print(f"Device Name: {dev_info['name']}")
        print(f"Trying to use this as Loopback Input...")
        # On Windows WASAPI, the loopback device is usually the output device itself opened as input
        print(f"SUGGESTED_DEVICE_ID = {default_output}")
    except Exception as e:
        print(e)
else:
    print("Could not find WASAPI host.")

print("\n--- ALL DEVICES ---")
print(sd.query_devices())
