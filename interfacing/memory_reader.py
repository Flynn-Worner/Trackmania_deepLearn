import time
from tminterface2 import TMInterface, MessageType

def run_reader():
    port = 8483
    iface = TMInterface(port)
    
    print(f"Attempting to connect to TMInterface on port {port}...")
    print("Waiting for Trackmania to connect via Python_Link.as...")
    
    try:
        iface.register(timeout=None)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("Connected! Listening for game states...")
    
    last_print = time.time()
    
    try:
        while True:
            # Read incoming message type from the game
            msgtype = iface._read_int32()
            
            if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                _time = iface._read_int32()
                
                # We fetch state and print it ~10 times a sec
                current_time = time.time()
                if current_time - last_print >= 0.1:
                    state = iface.get_simulation_state()
                    speed = state.display_speed
                    pos = state.position
                    print(f"Time: {_time}ms | Speed: {speed} km/h | Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
                    last_print = current_time
                    
                # Respond so the game doesn't freeze!
                iface._respond_to_call(msgtype)
                
            elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                current = iface._read_int32()
                target = iface._read_int32()
                iface._respond_to_call(msgtype)
                
            elif msgtype == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):
                iface._read_int32()
                iface._read_int32()
                iface._respond_to_call(msgtype)
                
            elif msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
                # Game just loaded in
                print("Game connected/restarted map!")
                # Tell it to update us every 10 steps (so it's faster)
                iface.set_on_step_period(10)
                iface._respond_to_call(msgtype)
                
            elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                # Just in case it sends a frame we didn't ask for
                iface._respond_to_call(msgtype)
                
            else:
                # Always respond to sync messages to avoid hanging the game engine
                iface._respond_to_call(msgtype)
                
    except KeyboardInterrupt:
        print("\nClosing connection...")
        try:
            iface.close()
        except:
            pass
        print("Disconnected.")

if __name__ == '__main__':
    run_reader()
