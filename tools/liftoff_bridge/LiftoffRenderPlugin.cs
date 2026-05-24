using BepInEx;
using UnityEngine;

namespace DroneInterception.LiftoffBridge
{
    [BepInPlugin("drone-interception.liftoff-render-bridge", "Liftoff Render Bridge", "0.1.0")]
    public sealed class LiftoffRenderPlugin : BaseUnityPlugin
    {
        private LiftoffRenderServer server;

        private void Awake()
        {
            DontDestroyOnLoad(gameObject);
            server = gameObject.AddComponent<LiftoffRenderServer>();
            server.Logger = Logger;
            server.Port = 47391;
            server.StartServer();
        }

        private void OnDestroy()
        {
            Logger.LogInfo("Liftoff render bridge plugin destroying");
            if (server != null)
            {
                server.StopServer();
            }
        }
    }
}
