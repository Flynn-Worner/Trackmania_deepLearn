Net::Socket@ socket;

uint counter = 0;
string currentAction = "MANUAL";

void Main()
{
    log("TMNF bridge plugin started.");

    @socket = Net::Socket();

    socket.Connect("127.0.0.1", 9000, 1000);

    log("Socket connection attempted.");
}

void ClearControls(SimulationManager@ simManager)
{
    simManager.SetInputState(InputType::Up, 0);
    simManager.SetInputState(InputType::Down, 0);
    simManager.SetInputState(InputType::Left, 0);
    simManager.SetInputState(InputType::Right, 0);
}

void ApplyAction(SimulationManager@ simManager, string action)
{
    if (action == "MANUAL")
    {
        return;
    }

    if (action == "ENTER" || action == "RESPAWN")
    {
        ClearControls(simManager);

        // ENTER token is treated as restart for compatibility with Python side.
        simManager.Respawn();

        currentAction = "MANUAL";

        log("Enter-style restart requested from Python.");

        return;
    }

    if (action == "FINISHRESET")
    {
        ClearControls(simManager);

        // Use simulation APIs directly to recover even when finish UI blocks input.
        simManager.GiveUp();
        simManager.Respawn();

        currentAction = "MANUAL";

        log("Direct finish reset requested from Python.");

        return;
    }

    if (action == "HARDSTOP")
    {
        ClearControls(simManager);

        auto dyna = simManager.get_Dyna();
        dyna.CurrentState.LinearSpeed = vec3(0, 0, 0);

        currentAction = "MANUAL";

        log("Hard stop requested from Python (velocity forced to zero).");

        return;
    }

    if (action == "GIVEUP" || action == "RESET")
    {
        ClearControls(simManager);

        simManager.GiveUp();

        currentAction = "MANUAL";

        log("GiveUp requested from Python.");

        return;
    }

    ClearControls(simManager);

    if (action == "ACCEL")
    {
        simManager.SetInputState(InputType::Up, 1);
    }
    else if (action == "LEFT")
    {
        simManager.SetInputState(InputType::Left, 1);
        simManager.SetInputState(InputType::Up, 1);
    }
    else if (action == "RIGHT")
    {
        simManager.SetInputState(InputType::Right, 1);
        simManager.SetInputState(InputType::Up, 1);
    }
    else if (action == "COAST")
    {
        // ClearControls already released everything
    }
}

void OnRunStep(SimulationManager@ simManager)
{
    counter++;

    auto dyna = simManager.get_Dyna();

    vec3 pos = dyna.CurrentState.Location.Position;
    vec3 vel = dyna.CurrentState.LinearSpeed;

    float yaw, pitch, roll;
    dyna.CurrentState.Location.Rotation.GetYawPitchRoll(yaw, pitch, roll);

    float speed = vel.Length();

    if (counter % 10 == 0)
    {
        string msg =
            "" + pos.x + "," +
            pos.y + "," +
            pos.z + "," +
            speed + "," +
            yaw + "\n";

        socket.Write(msg);

        string action = socket.ReadString(16);

        if (action.Length > 0)
        {
            currentAction = action;
            log("Action from Python: " + currentAction);
        }
    }

    ApplyAction(simManager, currentAction);
}

void OnDisabled()
{
    log("TMNF bridge plugin disabled.");
}

PluginInfo@ GetPluginInfo()
{
    auto info = PluginInfo();

    info.Name = "TMNF Python Bridge";
    info.Author = "Ryan";
    info.Version = "v1.0.0";
    info.Description = "Sends telemetry to Python and applies Python control actions.";

    return info;
}