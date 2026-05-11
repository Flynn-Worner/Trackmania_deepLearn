/**
 * Python Link – Openplanet plugin for TMInterface 2.1.x
 *
 * Sets up a TCP socket server so Python's TrackmaniaEnv can drive the car.
 *
 * =========================================================================
 * PORT CONFIGURATION FOR MULTIPLE TM INSTANCES
 * =========================================================================
 *
 * METHOD 1 – AUTO-SCAN (easiest, no configuration needed):
 *   This plugin automatically tries ports 8483, 8484, 8485 … 8492 and
 *   binds to the first one that is free.  If you open three TM windows,
 *   instance 1 gets 8483, instance 2 gets 8484, instance 3 gets 8485.
 *   Python side: run  python main.py --ports 8483 8484 8485
 *   NOTE: start ALL TM instances before running main.py, so every port
 *   is bound before Python tries to connect.
 *
 * METHOD 2 – scripts/launch_multienv.ps1 (sequential auto-patching):
 *   Patches the plugin with a specific port, launches TM, waits, repeats.
 *   Use this only if the try/catch auto-scan does not work on your build.
 *
 * See MULTI_INSTANCE_GUIDE.md for full instructions.
 * =========================================================================
 */

Net::Socket@ sock = null;
Net::Socket@ clientSock = null;

enum MessageType {
    SCRunStepSync = 1,
    SCCheckpointCountChangedSync = 2,
    SCLapCountChangedSync = 3,
    SCRequestedFrameSync = 4,
    SCOnConnectSync = 5,
    CSetSpeed = 6,
    CRewindToState = 7,
    CRewindToCurrentState = 8,
    CGetSimulationState = 9,
    CSetInputState = 10,
    CGiveUp = 11,
    CPreventSimulationFinish = 12,
    CShutdown = 13,
    CExecuteCommand = 14,
    CSetTimeout = 15,
    CRaceFinished = 16,
    CRequestFrame = 17,
    CResetCamera = 18,
    CSetOnStepPeriod = 19,
    CUnrequestFrame = 20,
    CToggleInterface = 21,
    CIsInMenus = 22,
    CGetInputs = 23,
}

const bool debug = false;
const string HOST = "127.0.0.1";
uint16 PORT;
uint RESPONSE_TIMEOUT = 2000;
int next_frame_requested_H = -1;
int next_frame_requested_W = -1;
int on_step_period = 10;
bool on_connect_queued = false;
auto@ simManager = GetSimulationManager();

void Init_Socket(){
    if (@sock is null) {
        @sock = Net::Socket();
        log("Port set to " + PORT);
        sock.Listen(HOST, PORT);
    }
}

void close_connection(){
    @clientSock = null;
    Init_Socket();
    next_frame_requested_H = -1;
    RESPONSE_TIMEOUT = 2000;
}

void WaitForResponse(MessageType type){
    auto now = Time::Now;

    while (true) {
        auto receivedType = HandleMessage();
        if (receivedType == type) {
            break;
        }

        if (receivedType == MessageType::CShutdown) {
            break;
        }

        if (receivedType == -1 && Time::Now - now > RESPONSE_TIMEOUT) {
            log("Client disconnected due to timeout (" + RESPONSE_TIMEOUT + "ms)");
            close_connection();
            break;
        }
    }
}

int HandleMessage()
{
    if (clientSock.Available == 0) {
        return -1;
    }

    int type = clientSock.ReadInt32();
    switch(type) {
        case MessageType::SCRunStepSync: {
            break;
        }

        case MessageType::SCRequestedFrameSync: {
            break;
        }

        case MessageType::SCCheckpointCountChangedSync: {
            break;
        }

        case MessageType::SCOnConnectSync: {
            break;
        }

        case MessageType::CSetSpeed: {
            if(debug){
                print("Server: SetSpeed message");
            }
            const float new_speed = clientSock.ReadFloat();
            simManager.SetSpeed(new_speed);
            if(debug){
                print("Server: Set speed to "+new_speed);
            }
            break;
        }

        case MessageType::CGiveUp: {
            if(debug){
                print("Server: Give up");
            }
            if (simManager.InRace) {
                simManager.GiveUp();
            }
            break;
        }

        case MessageType::CPreventSimulationFinish: {
            if(debug){
                print("Server: prevent simulation finish");
            }
            if (simManager.InRace) {
                simManager.PreventSimulationFinish();
            }
            break;
        }

        case MessageType::CRewindToState: {
            const int32 stateLength = clientSock.ReadInt32();
            const auto stateData = clientSock.ReadBytes(stateLength);
            if(debug){
                print("Server: rewind message");
            }
            if (simManager.InRace) {
                SimulationState state(stateData);
                simManager.RewindToState(state);
            }
            break;
        }

        case MessageType::CRewindToCurrentState: {
            if (simManager.InRace) {
                simManager.RewindToState(simManager.SaveState());
            }
            break;
        }

        case MessageType::CGetSimulationState: {
            auto@ state = simManager.SaveState();
            const auto@ data = state.ToArray();
            if(debug){
                print("Server: get_simulation_state");
            }

            clientSock.Write(int(data.Length));
            clientSock.Write(data);
            break;
        }

        case MessageType::CSetInputState: {
            if(debug){
                print("Server: Set input state message");
            }
            const bool accelerate = clientSock.ReadUint8()>0;
            const bool brake = clientSock.ReadUint8()>0;
            const int32 steer = clientSock.ReadInt32();

            if(debug){
                print("Set input state: accel="+accelerate+" brake="+brake+" steer="+steer);
            }

            if (simManager.InRace) {
                simManager.SetInputState(InputType::Up, accelerate?1:0);
                simManager.SetInputState(InputType::Down, brake?1:0);
                // Steer is an integer in [-65536, 65536].  Negative = left, positive = right.
                simManager.SetInputState(InputType::Left, steer < 0 ? -steer : 0);
                simManager.SetInputState(InputType::Right, steer > 0 ? steer : 0);
            }

            break;
        }

        case MessageType::CShutdown: {
            log("Client disconnected");
            close_connection();
            break;
        }

        case MessageType::CExecuteCommand: {
            const int32 bytes_to_read = clientSock.ReadInt32();
            const string command = clientSock.ReadString(bytes_to_read);
            if(debug){
                print("Server: command "+command+" received");
            }
            ExecuteCommand(command);
            break;
        }

        case MessageType::CSetTimeout: {
            const uint new_timeout = clientSock.ReadUint32();
            if(debug){
                print("Server: set timeout to "+new_timeout);
            }
            RESPONSE_TIMEOUT = new_timeout;
            break;
        }

        case MessageType::CRaceFinished: {
            const int is_race_finished = ((simManager.PlayerInfo.RaceFinished || simManager.TickTime>simManager.RaceTime)?1:0);
            if(debug){
                print("Server: Answering race_finished with "+is_race_finished);
            }
            clientSock.Write(is_race_finished);
            break;
        }

        case MessageType::CRequestFrame: {
            next_frame_requested_W = clientSock.ReadInt32();
            next_frame_requested_H = clientSock.ReadInt32();
            if(debug){
                print("Client requested next frame in size "+next_frame_requested_H+" "+next_frame_requested_W);
            }
            break;
        }

        case MessageType::CResetCamera: {
            simManager.ResetCamera();
            break;
        }

        case MessageType::CSetOnStepPeriod: {
            on_step_period = clientSock.ReadInt32();
            break;
        }

        case MessageType::CUnrequestFrame: {
            next_frame_requested_H = -1;
            break;
        }

        case MessageType::CToggleInterface: {
            const bool new_val = clientSock.ReadInt32()>0;
            ToggleRaceInterface(new_val);
            break;
        }

        case MessageType::CIsInMenus: {
            const int response = GetCurrentGameState()==TM::GameState::Menus? 1 : 0;
            clientSock.Write(response);
            break;
        }

        case MessageType::CGetInputs: {
            TM::InputEventBuffer@ inputs = simManager.get_InputEvents();
            const string input_text = inputs.ToCommandsText();
            clientSock.Write(int32(input_text.Length));
            clientSock.Write(input_text);
            break;
        }

        default: {
            log("Server: got unknown message "+type);
            break;
        }
    }

    return type;
}

void OnRunStep(SimulationManager@ simManager){
    if (@clientSock is null) {
        return;
    }
    if(debug){
        print("Server: OnRunStep " + simManager.RaceTime);
    }

    if(simManager.RaceTime%on_step_period==0 || simManager.TickTime>simManager.RaceTime){
        clientSock.Write(MessageType::SCRunStepSync);
        clientSock.Write(simManager.RaceTime);
        WaitForResponse(MessageType::SCRunStepSync);
    }
}

void OnCheckpointCountChanged(SimulationManager@ simManager, int current, int target){
    if (@clientSock is null) {
        return;
    }
    if(debug){
        print("Server: OnCheckpointCountChanged");
    }

    clientSock.Write(MessageType::SCCheckpointCountChangedSync);
    clientSock.Write(current);
    clientSock.Write(target);
    WaitForResponse(MessageType::SCCheckpointCountChangedSync);
}

void OnLapCountChanged(SimulationManager@ simManager, int current, int target){
    if (@clientSock is null) {
        return;
    }
    if(debug){
        print("Server: OnLapCountChanged");
    }

    clientSock.Write(MessageType::SCLapCountChangedSync);
    clientSock.Write(current);
    clientSock.Write(target);
    WaitForResponse(MessageType::SCLapCountChangedSync);
}

void OnConnect(){
    clientSock.Write(MessageType::SCOnConnectSync);
    WaitForResponse(MessageType::SCOnConnectSync);
}

/**
 * Try to bind a specific port.
 * Returns true on success, false if the port is already taken.
 *
 * Net::Socket::Listen() returns bool (true = bound OK, false = port in use).
 * We MUST check this return value.  A try/catch alone is NOT sufficient
 * because Listen() signals failure via its return value, not an exception.
 * Ignoring the return value means every instance silently claims port 8483.
 */
bool TryBindPort(uint16 p)
{
    @sock = Net::Socket();
    if (sock.Listen(HOST, p)) {
        PORT = p;
        return true;
    }
    @sock = null;
    return false;
}

/**
 * Main() – pure auto-scan, no RegisterVariable / CommandList.
 *
 * WHY no RegisterVariable:
 *   TMInterface persists variable values between sessions to disk.
 *   The old OnQueueProcessed pattern relied on RegisterVariable("custom_port"),
 *   but the CommandList fires ASYNCHRONOUSLY (after Main() completes), so it
 *   ran AFTER the auto-scan had already claimed a port.  Worse, if a previous
 *   session had written a value like 8484 to the variable store, every new
 *   instance would read that stale value and re-bind to 8484, overriding the
 *   scan result — producing the bug where every instance claimed port 8484.
 *
 * HOW multi-instance works now:
 *   Open TM windows one at a time.  Each calls Main() and tries 8483 first.
 *   Instance 1 succeeds → binds 8483.
 *   Instance 2 finds 8483 taken → tries 8484 → succeeds.
 *   Instance 3 → 8485, and so on up to 8492.
 *   Python side: python main.py --ports 8483 8484 8485
 */
void Main()
{
    bool bound = false;
    for (uint16 p = 8483; p <= 8492; p++) {
        if (TryBindPort(p)) {
            log("Python Link: listening on port " + PORT);
            bound = true;
            break;
        }
        log("Python Link: port " + p + " busy, trying " + (p + 1) + "...");
    }

    if (!bound) {
        log("Python Link ERROR: no free port in 8483-8492. "
            "Close other applications using those ports and reload the plugin.");
    }
}

void OnGameStateChanged(TM::GameState state){
    if(state == TM::GameState::Menus && on_connect_queued){
        OnConnect();
        on_connect_queued = false;
    }
}

void Render(){
    //Bluescreens if you print every Render()
    auto @newSock = sock.Accept(0);
    if (@newSock !is null) {
        @clientSock = @newSock;
        newSock.NoDelay = true;
        log("Client connected (IP: " + clientSock.RemoteIP + ")");
        if(GetCurrentGameState() != TM::GameState::StartUp){
            OnConnect();
        }
        else{
            on_connect_queued = true;
        }
    }
    if(next_frame_requested_H>=0){
        const auto@ frame = Graphics::CaptureScreenshot(vec2(next_frame_requested_W,next_frame_requested_H));
        clientSock.Write(MessageType::SCRequestedFrameSync);
        clientSock.Write(frame);
        WaitForResponse(MessageType::SCRequestedFrameSync);
        next_frame_requested_H = -1;
        if(debug){
            print("Got response from client for SCRequestedFrameSync");
        }
    }
}

PluginInfo@ GetPluginInfo(){
    PluginInfo info;
    info.Author = "Agade";
    info.Name = "Python Link";
    info.Description = "Reproduce close to original TMI <2 python interface with TMI 2.1 sockets";
    info.Version = "0.1";
    return info;
}
