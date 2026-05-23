using BepInEx.Logging;
using System;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace DroneInterception.LiftoffBridge
{
    public sealed class LiftoffRenderServer : MonoBehaviour
    {
        public ManualLogSource Logger { get; set; }
        public int Port { get; set; }

        private readonly ConcurrentQueue<RenderJob> jobs = new ConcurrentQueue<RenderJob>();
        private TcpListener listener;
        private Thread serverThread;
        private volatile bool running;
        private LiftoffCameraBinder cameraBinder;

        private void Awake()
        {
            Port = 47391;
        }

        public void StartServer()
        {
            if (running)
            {
                return;
            }

            cameraBinder = gameObject.AddComponent<LiftoffCameraBinder>();
            running = true;
            listener = new TcpListener(IPAddress.Loopback, Port);
            listener.Start();
            serverThread = new Thread(ServerLoop) { IsBackground = true };
            serverThread.Start();
            if (Logger != null)
            {
                Logger.LogInfo("Liftoff render bridge listening on 127.0.0.1:" + Port);
            }
        }

        public void StopServer()
        {
            running = false;
            if (Logger != null)
            {
                Logger.LogInfo("Liftoff render bridge stopping");
            }
            if (listener != null)
            {
                listener.Stop();
            }
        }

        private void Update()
        {
            RenderJob job;
            while (jobs.TryDequeue(out job))
            {
                try
                {
                    RenderFrameResponse response = cameraBinder.Render(job.Request);
                    job.Complete(response, null);
                }
                catch (Exception ex)
                {
                    job.Complete(null, ex);
                }
            }
        }

        private void ServerLoop()
        {
            while (running)
            {
                try
                {
                    TcpClient client = listener.AcceptTcpClient();
                    ThreadPool.QueueUserWorkItem(delegate { HandleClient(client); });
                }
                catch (SocketException)
                {
                    if (running && Logger != null)
                    {
                        Logger.LogError("Liftoff render bridge socket failure");
                    }
                }
                catch (Exception ex)
                {
                    if (running && Logger != null)
                    {
                        Logger.LogError("Liftoff render bridge server failure: " + ex);
                    }
                }
            }
        }

        private void HandleClient(TcpClient client)
        {
            using (client)
            using (NetworkStream stream = client.GetStream())
            {
                byte[] header = ReadExact(stream, 8);
                int payloadLength = BitConverter.ToInt32(header, 0);
                byte[] payload = ReadExact(stream, payloadLength);
                string json = Encoding.UTF8.GetString(payload);

                RenderFrameRequest request = RenderProtocol.ParseRequest(json);
                RenderJob job = new RenderJob(request);
                jobs.Enqueue(job);
                RenderFrameResponse response = job.Wait();

                byte[] frameInfo = Encoding.UTF8.GetBytes(RenderProtocol.FrameInfoJson(response));
                byte[] outHeader = new byte[8];
                Buffer.BlockCopy(BitConverter.GetBytes(frameInfo.Length), 0, outHeader, 0, 4);
                Buffer.BlockCopy(BitConverter.GetBytes(response.Pixels.Length), 0, outHeader, 4, 4);
                stream.Write(outHeader, 0, outHeader.Length);
                stream.Write(frameInfo, 0, frameInfo.Length);
                stream.Write(response.Pixels, 0, response.Pixels.Length);
            }
        }

        private static byte[] ReadExact(NetworkStream stream, int length)
        {
            byte[] buffer = new byte[length];
            int offset = 0;
            while (offset < length)
            {
                int read = stream.Read(buffer, offset, length - offset);
                if (read <= 0) throw new InvalidOperationException("Client disconnected");
                offset += read;
            }
            return buffer;
        }
    }
}
