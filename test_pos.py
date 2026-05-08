from tminterface.client import Client, run_client
from interfacing.tminterface2 import TMInterface, MessageType
import time

def main():
    iface = TMInterface(8483)
    print("Connecting...")
    iface.register(timeout=None)
    
    while True:
        msgtype = iface._read_int32()
        if msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
            iface.set_on_step_period(10)
            iface._respond_to_call(msgtype)
            break
        else:
            iface._respond_to_call(msgtype)
            
    print("Connected. Waiting for step...")
    while True:
        msgtype = iface._read_int32()
        if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
            _time = iface._read_int32()
            state = iface.get_simulation_state()
            print(f"Time: {_time}, Position: {state.position}")
            iface._respond_to_call(msgtype)
            break
        else:
            iface._respond_to_call(msgtype)
            
    iface.close()

if __name__ == "__main__":
    main()
